# Work Plan 08 — OpenSAE brand identity

The three public surfaces now use the approved PNG bytes. The square organization avatar was not replaced: Hugging Face has no avatar-upload route for this token.

## Approved assets

| File | Pixels | Bytes | SHA-256 |
| --- | --- | --- | --- |
| `assets/branding/opensae_banner.png` | 2047 × 639 | 1,127,590 | `f825c27d130ba96ce5fb01112ff307707e58df9274577c4078483cc89ae7f773` |
| `assets/branding/opensae_symbol.png` | 1126 × 1087 | 56,602 | `a2b7f4f8326a2819c726dc0888f50b7c051f2f9ad06bb09e5189a7fd82f67ce0` |

Both files already matched these checksums on `origin/main` at `85cd805`. Published copies were downloaded again and matched the same hashes. The pixels were not redrawn, recolored, or cropped.

## Task A — organization card

The green graphic on https://huggingface.co/OpenSAE is the static Space `OpenSAE/README` (`index.html`), not the SAEBase Space and not the profile avatar.

- Header and the former green hero both use `opensae_banner.png`.
- The hero container is white, so the banner’s own light background is not a white block on green. The PNG pixels are unchanged.
- Image URL, which resolves when the card is inlined on `huggingface.co`: `https://huggingface.co/spaces/OpenSAE/README/resolve/main/assets/opensae_banner.png` (HTTP 302 to the CDN, body SHA-256 matches).
- Hugging Face drops `@media` rules when it inlines the card. The hero is a wrapping flex row, so a narrow card stacks the banner under the headline instead of covering the text.
- Space revision: `f1c764125ea3a75fac8c6f475ed1cdd48882356f`.

The square profile avatar still shows the same geometric mark, but the file is a 200×200 WebP (`4787b84d8606942056535d4e9de68c8da226632146eb92b11b988a55c7e92204`), not `opensae_symbol.png`. `POST /api/organizations/OpenSAE/avatar` and `POST /api/settings/organizations/OpenSAE/avatar` both return 404, and `huggingface_hub` exposes no organization-avatar method. The horizontal wordmark was not forced into that square slot.

## Task B — Research Lab header

Every module uses `opensae_symbol.png` at 40×38.6 px on a 1440 px desktop and 35×33.8 px at 390 px. The image is `object-fit: contain`, `alt="OpenSAE"`, and the home link is unchanged. The word “OpenSAE” beside it is the existing text name. There is no second drawn wordmark.

| Module | Public URL |
| --- | --- |
| Feature Atlas | https://opensae-saebase.static.hf.space/index.html#features |
| Feature Effects | https://opensae-saebase.static.hf.space/lab-experiments/index.html#effects |
| Context Lab | https://opensae-saebase.static.hf.space/lab-experiments/index.html#context |
| Compare models | https://opensae-saebase.static.hf.space/lab-experiments/index.html#models |
| Research | https://opensae-saebase.static.hf.space/lab-experiments/index.html#interactions |

Stylesheet cache key is `?v=wp08`. The served symbol is 56,602 bytes and matches the approved SHA-256. No image request returned 4xx. Space revision: `6d7c576f592f4749feec380216672c9c26f5cd5a`.

Feature Effects still opens on Gemma 1B, “The capital of France is”, feature #203, top token “ a” (23.0%). Turning it off still reports: “Turning off feature #203 changed the model's top next-token prediction from “ a” to “ Paris” on this input.”

## Task C — dataset card

`OpenSAE/SAEBase-Evaluation` README revision `e3d1f1c2c72defceff5d9e116bd1393f2a967c4d`.

The banner is the first image in the README body, immediately after the YAML metadata, centered, width 720, original aspect. The next heading is “SAEBase Evaluation”. Provenance, license, usage, and links are unchanged. Source pinned to the verified commit:

`https://raw.githubusercontent.com/OpenSAE/SAEBase/85cd805/assets/branding/opensae_banner.png`

Public page: https://huggingface.co/datasets/OpenSAE/SAEBase-Evaluation (Dataset card tab). The rendered image is 2047×639 from that URL.

## Task D — logo audit

Graphic placeholders replaced:

- Research Lab header glyph `✳` in the live Atlas and lab-experiment shells.
- Unused `apps/research-lab/frontend` hexagon `⬡`, pointed at the same symbol. That frontend is not the live Space.

Left in place: research-card SVGs on the organization card (section icons for SAR, RSS, and related studies) and scientific figures. No new logo was drawn.

## Repository files

- `apps/research-lab/lab-experiments/index.html`
- `apps/research-lab/lab-experiments/assets/lab.css`
- `apps/research-lab/space-overlay/index.html`
- `apps/research-lab/space-overlay/assets/style.css`
- `apps/research-lab/space-overlay/assets/brand/opensae_symbol.png`
- `apps/research-lab/frontend/index.html`
- `apps/research-lab/frontend/styles.css`

`features.json` was not uploaded. `assets/effect-links.json` was not changed.

## Screenshots

Before, organization card only (captured before this change):

- `outputs/research_lab_brand_08/org-before-desktop.png`
- `outputs/research_lab_brand_08/org-before-mobile.png`

After:

- `outputs/research_lab_brand_08/org-after-desktop.png`
- `outputs/research_lab_brand_08/org-after-mobile.png`
- `outputs/research_lab_brand_08/dataset-card-banner-desktop.png`
- `outputs/research_lab_brand_08/dataset-card-banner-mobile.png`
- `outputs/research_lab_brand_08/public-{atlas,effects,context,models,research}-{desktop,mobile}.png`

## Blocker

The organization profile avatar cannot be updated through the API available to this account. The wide organization card and the dataset opening use the approved banner. The Lab header uses the approved symbol.
