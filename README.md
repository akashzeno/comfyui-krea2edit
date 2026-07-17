# ComfyUI-Krea2Edit

Instruction-based image editing for **Krea 2** in ComfyUI — the node pack that powers
the **Krea 2 Identity Edit** LoRA. Turns Krea 2 (Raw or Turbo) into an image editor with dual
conditioning: the source image is injected both as VAE latent tokens (appearance) and
into the Qwen3-VL text encoder (semantic grounding), matching how the LoRA was trained.

## Model versions

See [CHANGELOG.md](CHANGELOG.md) — **v1.2 is recommended** (better face likeness,
plus the new `fit` reference geometry and `ref_boost` fidelity dial).

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lbouaraba/comfyui-krea2edit
# restart ComfyUI
```

Requirements: a ComfyUI version with native Krea 2 support, the Krea 2 model
(Raw or Turbo), the Qwen3-VL 4B text encoder used by Krea 2, and the Krea 2 Identity Edit
LoRA (`krea2_identity_edit_v1_2.safetensors`). No extra Python dependencies.

## Nodes

### `Krea2EditModelPatch`
Wraps the diffusion model so the VAE-encoded source image is prepended as clean
in-context tokens (RoPE frame 1). Inputs:
- `model` — Krea 2 (LoRA already applied)
- `source_latent` — VAEEncode of the image being edited
- `source_latent_b` *(optional)* — second reference (RoPE frame 2) for two-input
  edits (e.g. person + scene)
- `vae` + `source_image` *(optional, recommended)* — the blur-proof pixel path: give
  the raw image (and VAE) and the node fits it to the target grid in pixel space.
  Required for `fit_mode: fit`.
- `fit_mode` *(default `fit`)* — how a source fits a mismatched output aspect ratio.
  `fit` = training-matched resample at a centered offset (v1.2); `crop` = center-crop,
  the v1/v1.1-legacy geometry (use with older weights).
- `ref_boost` *(default 1.0)* — reference-fidelity dial; >1 pulls harder toward the
  reference's appearance, <1 loosens. `ref_boost_a` is the same dial for the scene ref in two-ref edits.

### `Krea2EditGroundedEncode`
Image-grounded instruction encoding — the text encoder *sees* the image while
reading your instruction, exactly as during training. Inputs:
- `clip` — the Krea 2 CLIP (Qwen3-VL, loaded with `type: krea2`)
- `prompt` — the edit instruction ("recolor the car to matte black")
- `image` — the same source image
- `image_b` *(optional)* — second reference for two-input edits
- `grounding_px` — grounding resolution (default 768; trained range 512–1536).
  This is a quality dial: lower = stronger edit adherence, higher = stronger
  identity/likeness. Try 1024+ for people, 512 for stubborn scene changes.

**Both nodes are required.** With a stock `CLIPTextEncode` the model never sees the
image semantically and quality drops sharply, especially for scene-referential
instructions ("the man on the left").

## Minimal wiring

```
LoadImage ─┬─ VAEEncode ── Krea2EditModelPatch.source_latent
           └─ Krea2EditGroundedEncode.image     (+ your prompt)
UNETLoader ── LoraLoaderModelOnly (krea2_identity_edit_v1_2 @1.0) ── Krea2EditModelPatch.model
Krea2EditModelPatch ── KSampler.model
Krea2EditGroundedEncode ── KSampler.positive
Krea2EditGroundedEncode (empty prompt, same image) ── KSampler.negative
EmptySD3LatentImage ── KSampler.latent_image
```

Example workflow in `workflows/`: `krea2_identity_edit.json` — single-image editor by
default; enable group 2 (toggle its Bypass off) for two-image person-into-scene edits.

## Usage notes (read these — they matter)

1. **Aspect ratio.** With `fit_mode: fit` (default in v1.2) and `vae` + `source_image`
   connected, mismatched source/output aspect ratios are handled — the source is
   resampled to the target grid. On `crop`/legacy weights, still match the AR: a
   mismatched AR is out of distribution and degrades identity/preservation.
2. **Turbo, 8 steps, CFG 1** is the fast path (~1 min at 2MP) and works for most
   edits: recolor, add/insert, attribute changes, restyles, scene translation.
3. **Removals and other "delete salient content" edits need real guidance:**
   use the **Raw** model at **CFG 3, ~20 steps**. Distilled Turbo at CFG 1 will
   usually re-render the subject instead of removing it.
4. At CFG > 1, ground the negative too: a second `Krea2EditGroundedEncode` with an
   empty prompt and the same image (this is the trained unconditional).
5. Two-input edits: scene image → `source_latent`/`image`, subject image →
   `source_latent_b`/`image_b`. Leave the b-inputs unconnected for single-image use.
6. **Generate at ≤2MP.** Above the trained range, source content can bleed into
   the output or subjects duplicate.
7. **Two people with distinct faces:** chain single-ref inserts (place person A,
   then run a second edit adding person B from their reference) — currently more
   face-faithful than one two-ref pass.

## License / credits

Nodes: Apache-2.0. The **Krea 2 Identity Edit** weights ship separately under the
Krea 2 Community License Agreement (see the model card, `LICENSE.pdf`, and `NOTICE`
in the weights repo).
Built on Krea 2 by Krea AI; text encoder Qwen3-VL (Alibaba).
