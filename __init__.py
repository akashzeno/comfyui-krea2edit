"""ComfyUI-Krea2Edit — in-context edit forward for the Krea2 model.

ComfyUI's native Krea2 `_forward` is text-to-image only: it builds the sequence
`[text | target]`. The krea2_edit LoRA (trained in ai-toolkit) needs the *appearance
path*: the VAE-encoded SOURCE latent prepended as a block of clean tokens, distinguished
from the (noisy) target purely by the 3-axis RoPE frame index (source=1, target=0, h/w
aligned). This node adds that by wrapping the model's DIFFUSION_MODEL forward and rebuilding
the sequence as `[text | source(frame=1) | target(frame=0)]`, keeping only the target tokens
out — mirroring ai-toolkit's `predict_velocity_edit` exactly, using the model's own submodules.

Wiring:  LoadImage -> VAEEncode(source) --\
                                            Krea2EditModelPatch(model, source_latent) -> KSampler
         UNETLoader -> LoraLoaderModelOnly -/
KSampler.latent_image <- EmptySD3LatentImage (noise). Text: NATIVE krea2 CLIP + CLIPTextEncode.
"""
import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.patcher_extension
import comfy.utils
import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding


def _imgids(bs, frame, h_, w_, device):
    ids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)


def _to_4d(v):
    """(B,C,T,H,W) -> (B*T,C,H,W); pass 4D through. Images use T=1."""
    if v.ndim == 5:
        b, c, t, h, w = v.shape
        return v.reshape(b * t, c, h, w)
    return v


def krea2_edit_forward(m, x, timesteps, context, src_latent, transformer_options):
    """Krea2 SingleStreamDiT._forward, but with source block(s) prepended.

    m           : the SingleStreamDiT (LoRA-patched at sample time)
    x           : (B,C,H,W) or (B,C,T,H,W) noisy TARGET latent
    src_latent  : clean SOURCE latent (VAE-encoded), 4D/5D — or a LIST of them
                  (multi-ref: [scene, subject], frames 1..N, training-matched)
    context     : (B, seq, txtlayers*txtdim) — the 12-layer Qwen3-VL stack
    """
    patch = m.patch

    # Mirror ComfyUI _forward: latents may arrive 5D (B,C,T,H,W) for this model.
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
    x = _to_4d(x)
    bs, c, H_orig, W_orig = x.shape

    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch

    # source(s) -> (bs, C, H, W): flatten temporal, match batch, resize to the target grid.
    src_list = src_latent if isinstance(src_latent, (list, tuple)) else [src_latent]
    srcs = []
    for sl in src_list:
        src = _to_4d(sl).to(x.device, x.dtype)
        if src.shape[0] != bs:
            src = src[:1].expand(bs, *src.shape[1:])
        if src.shape[-2:] != (H, W):
            src = F.interpolate(src.float(), size=(H, W), mode="bilinear").to(x.dtype)
        srcs.append(comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch)))

    context = m._unpack_context(context)                       # (B, seq, 12, 2560)

    tgt_img = m.first(rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
    src_imgs = [m.first(rearrange(s_, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
                for s_ in srcs]

    t = m.tmlp(timestep_embedding(timesteps, m.tdim).unsqueeze(1).to(tgt_img.dtype))
    tvec = m.tproj(t)

    context = m.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = m.txtmlp(context)

    txtlen, tgtlen = context.shape[1], tgt_img.shape[1]
    srclen = sum(si.shape[1] for si in src_imgs)
    combined = torch.cat([context] + src_imgs + [tgt_img], dim=1)  # [text | refs... | target]

    device = combined.device
    pos = torch.cat([
        torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)]   # text @ 0
        + [_imgids(bs, i + 1, h_, w_, device) for i in range(len(src_imgs))]  # refs frame=1..N
        + [_imgids(bs, 0, h_, w_, device)],                                    # target frame=0
        dim=1)
    freqs = m.pe_embedder(pos)

    for block in m.blocks:
        combined = block(combined, tvec, freqs, None, transformer_options=transformer_options)

    final = m.last(combined, t)
    out = final[:, txtlen + srclen: txtlen + srclen + tgtlen, :]         # target tokens only
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_, w=w_, ph=patch, pw=patch, c=m.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, m.channels, H_orig, W_orig).movedim(1, 2)
    return out


class Krea2EditModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "source_latent": ("LATENT",),
        }, "optional": {
            "source_latent_b": ("LATENT", {"tooltip": "2nd reference (subject photo) for multi-ref LoRAs -> RoPE frame=2, training-matched order: scene first, subject second"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "krea2edit"
    DESCRIPTION = "Adds the krea2_edit in-context source-preservation path (source latent as frame=1 tokens) to a Krea2 model."

    def patch(self, model, source_latent, source_latent_b=None):
        m = model.clone()
        # The target latent reaches the diffusion model already scaled (process_latent_in);
        # scale the source(s) the same way so all share one latent space.
        src_samples = model.model.process_latent_in(source_latent["samples"])
        if source_latent_b is not None:
            src_samples = [src_samples, model.model.process_latent_in(source_latent_b["samples"])]

        def wrapper(executor, x, timesteps, context, attention_mask=None, transformer_options={}, **kwargs):
            dm = executor.class_obj  # the SingleStreamDiT instance
            return krea2_edit_forward(dm, x, timesteps, context, src_samples, transformer_options)

        to = m.model_options.setdefault("transformer_options", {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "krea2_edit", wrapper, to
        )
        return (m,)


class Krea2EditGroundedEncode:
    """Image-grounded instruction encode — the SEMANTIC path of krea2_edit.

    Training always encodes the instruction TOGETHER with the source image through
    Qwen3-VL (user turn = <vision tokens: source> + instruction) and taps 12 layers.
    Stock CLIPTextEncode is text-only, so inference was running with the grounding
    half of the recipe missing (the VAE source tokens carry appearance; THIS carries
    scene semantics: "the man on the left", "the sign in the back").

    Requires a qwen3vl TE checkpoint WITH the vision tower (all local ones have it).
    grounding_px caps the longest side fed to the VLM — the 2026-07-02 LoRA trained
    with 384-768px jitter, so 640-768 is in-distribution; 0 = native res (the jitter
    training makes that tolerable too). For CFG, ground the NEGATIVE too: second node,
    empty prompt, same image (matches training's unconditional).
    """
    KREA2_EDIT_TEMPLATE = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "{}<|im_end|>\n<|im_start|>assistant\n"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_b": ("IMAGE", {"tooltip": "2nd reference (subject) for multi-ref LoRAs; vision blocks in training order: scene, subject"}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 64,
                                          "tooltip": "cap longest side fed to Qwen3-VL; 0 = native"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "krea2edit"
    DESCRIPTION = "Encodes the edit instruction grounded on the source image (training-matched semantic path)."

    KREA2_EDIT_TEMPLATE_2REF = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "{}<|im_end|>\n<|im_start|>assistant\n"
    )

    def _prep(self, image, grounding_px):
        samples = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
        h, w = samples.shape[2], samples.shape[3]
        if grounding_px and max(h, w) > grounding_px:
            s = grounding_px / max(h, w)
            samples = comfy.utils.common_upscale(samples, round(w * s), round(h * s), "area", "disabled")
        return samples.movedim(1, -1)[:, :, :, :3]

    def encode(self, clip, prompt, image=None, image_b=None, grounding_px=768):
        if image is None:  # text-only fallback = old behavior
            tokens = clip.tokenize(prompt)
            return (clip.encode_from_tokens_scheduled(tokens),)
        imgs = [self._prep(image, grounding_px)]
        template = self.KREA2_EDIT_TEMPLATE
        if image_b is not None:
            imgs.append(self._prep(image_b, grounding_px))
            template = self.KREA2_EDIT_TEMPLATE_2REF
        tokens = clip.tokenize(prompt, images=imgs, llama_template=template)
        return (clip.encode_from_tokens_scheduled(tokens),)


NODE_CLASS_MAPPINGS = {
    "Krea2EditModelPatch": Krea2EditModelPatch,
    "Krea2EditGroundedEncode": Krea2EditGroundedEncode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2EditModelPatch": "Krea2 Edit (source patch)",
    "Krea2EditGroundedEncode": "Krea2 Edit (grounded encode)",
}
