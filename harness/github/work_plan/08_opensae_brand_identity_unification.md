# Work Plan 08 — OpenSAE Brand Identity Unification

Created: 2026-10-20  
Status: Ready for execution. The two approved original PNG assets and this plan are committed to SAEBase. Changes to external Hugging Face pages and deployed Lab remain execution tasks, not claims of completed deployment.  
Scope: OpenSAE Hugging Face organization card, OpenSAE Research Lab top-left identity, and the opening of the SAEBase Evaluation dataset page.

## 0. Decision

There are exactly **two authorized OpenSAE graphic identities**:

- Full horizontal wordmark: `saebase/assets/branding/opensae_banner.png` (mark plus the word OpenSAE).
- Standalone mark: `saebase/assets/branding/opensae_symbol.png` (the geometric symbol only).

These PNGs are the **original user-supplied images, byte-for-byte**. Do not redraw, reinterpret, recreate with image generation, trace approximately, recolor, stretch, crop their essential content, apply arbitrary filters, or replace them with another OpenSAE logo. Normal textual use of the name OpenSAE in headings and navigation is allowed, but should not substitute for the visual identity where a logo belongs. No third graphic-logo variant.

**Original asset identity**

| Approved file | Dimensions | SHA-256 |
| --- | --- | --- |
| `saebase/assets/branding/opensae_banner.png` | 2047 × 639 | `f825c27d130ba96ce5fb01112ff307707e58df9274577c4078483cc89ae7f773` |
| `saebase/assets/branding/opensae_symbol.png` | 1126 × 1087 | `a2b7f4f8326a2819c726dc0888f50b7c051f2f9ad06bb09e5189a7fd82f67ce0` |

Use direct raw links if an external page cannot share a checked-out local asset:

- `https://raw.githubusercontent.com/OpenSAE/SAEBase/main/saebase/assets/branding/opensae_banner.png`
- `https://raw.githubusercontent.com/OpenSAE/SAEBase/main/saebase/assets/branding/opensae_symbol.png`

Pin by commit SHA or copy the verified bytes into the particular deployment's assets when a durable, reproducible release needs it; do not silently hotlink mutable branches without a check.

## 1. Task A — OpenSAE Hugging Face organization card

Page to inspect: `https://huggingface.co/OpenSAE`.

The organization card currently has a green-background graphical area. The approved OpenSAE banner must be its brand image, not an unrelated placeholder or the old geometric icon. Identify whether the requested region is rendered from the organization avatar, organization banner, README imagery, or another supported HF profile field. Use the real supported profile control or source asset for **that exact visible region**. Do not confuse the organization profile image with the unrelated OpenSAE Hugging Face Space, and do not claim a successful visual change solely because a README image changed.

Layout:

- Use the full horizontal `opensae_banner.png` for the larger card graphic wherever the profile control supports a suitable wide graphic. If the specific HF avatar slot is square-only, use `opensae_symbol.png` there and preserve the banner in the wide region that HF actually exposes. Do not mutilate the horizontal wordmark to force it into a square avatar.
- Maintain the original aspect ratio, centered in the available area, with meaningful padding. Avoid visible distortion or clipping.
- The supplied PNG already has a light background. If a green card/background would produce an awkward white rectangle, adjust only the **surrounding container/background when HF controls allow it**, not the supplied brand pixels; otherwise use the standalone symbol in square slots and put the banner in the appropriate wide field. Do not invent a newly recolored logo.
- No extra slogans, tiny descriptions, badges or duplicate logos added merely for decoration.
- If Hugging Face does not expose the relevant card image for customization, document the actual UI/platform limitation, change the nearest explicitly supported branded surface, and show screenshots. Do not pretend the green card itself was updated.

Acceptance: visually inspect the **actual public organization card**, not just profile settings. Record screenshot and effective source URL. Confirm the displayed mark/wordmark originated from these approved images.

## 2. Task B — Research Lab top-left identity

Live destination: `https://opensae-saebase.static.hf.space/index.html` (use current build/version, not a stale query string).

- Identify the actual deployed frontend source; GitHub `saebase/research_lab/frontend/` may lag newer HF Space code. Preserve the working Feature Atlas and Work Plans 06/07 behavior.
- Replace any placeholder hexagon, emoji, CSS-painted logo, generic icon or inconsistent top-left brand symbol with `opensae_symbol.png`. If sufficient desktop room exists, the original full banner may replace it instead, at natural aspect ratio, but do not combine the symbol with a separately styled faux OpenSAE logo.
- Use one shared header component across Feature Atlas, Feature Effects, Context Lab, Compare Models and Research. Ensure it has the SAME brand asset, size, padding and alignment on all routes.
- Keep header accessible: `alt="OpenSAE"`, visible keyboard focus and working home link. Check that the mark is crisp without stretching, and that narrow screens do not clip header or navigation.
- Do not disturb the first Lab's approved scientific content, navigation state or typography. Follow Work Plan 07 for shared layout unification.

Acceptance: screenshot left header on all five Lab modules at 1440px desktop and 390px mobile; browser asset requests must resolve correctly, and navigation must not change the logo or its position.

## 3. Task C — SAEBase Evaluation dataset opening

Target dataset: `https://huggingface.co/datasets/OpenSAE/SAEBase-Evaluation`.

- Edit the **existing dataset README / dataset card** rather than creating a second dataset or replacing its scientific description.
- Place the approved full `opensae_banner.png` prominently at the beginning of the rendered README content, directly below required Hugging Face YAML metadata if it exists. Use a correct Markdown/HTML image reference to a reachable, versioned source.
- Preserve the original dataset's title, data/provenance explanations, licensing, dataset usage, links and meaningful material. Only replace old/unrelated brand imagery and remove duplicate decorative headings where necessary.
- Responsive presentation: max-width is appropriate for the page; keep original proportions, centered with modest whitespace. Do not add tiny commentary or a second logo immediately beneath it.
- Distinguish the **dataset card top** from the HF organization profile/card. Both must be separately verified.

Acceptance: open the actual public dataset page and confirm that the first brand visual after dataset metadata is the full OpenSAE banner, with no broken image URLs or redundant old logo.

## 4. Task D — one global brand usage audit

Scan the actual source repositories and connected published surfaces that contain OpenSAE brand images, including:

- SAEBase asset references, README and Lab frontend;
- the OpenSAE organization/profile README and any owned HF Space branding components;
- the SAEBase Evaluation dataset README and references;
- legacy placeholders such as `⬡`, old hexagon components, custom faux logo glyphs and unrelated banner paths.

Replace **graphic logo usages** only when they are in scope and have a safe, verified route to the same approved assets. Do NOT replace unrelated scientific figures, charts, screenshots, dataset artwork or icons that are not being used as OpenSAE logos. Avoid blanket string replacements that break examples or docs. Use one source of truth for full/banner vs standalone/symbol.

No image-generation tool, icon redesign, third candidate mark or alternative handwritten OpenSAE wordmark. Do not introduce new variants without user approval.

## 5. Engineering and deployment procedure

1. Verify the two files already in SAEBase `saebase/assets/branding/` against the SHA-256 values above. If mismatch, restore from the approved committed originals; do not use similarly named but different assets.
2. Inspect the current HF organization control/page, actual Lab frontend and dataset README; save before screenshots.
3. Update the three requested surfaces, using the banner for the organization card's wide graphic and dataset opening, and the symbol for the Lab's compact header.
4. Keep modifications minimal and isolated. Avoid changing the evaluation dataset's content, the Lab data records, the first Atlas experience or experimental claims.
5. Preview at desktop and mobile, including the HF organization card and dataset landing.
6. Commit/submit through the appropriate GitHub/HF repositories with current authenticated access. Do not claim access to HF profile modification if connector permissions do not allow it; record a truthful blocker and continue with the other surfaces.
7. After deployment, hard-reload the three public destinations, confirm image source URLs and inspect screenshots. Check the exact source asset bytes from deployment against approved SHA-256 where the bytes can be downloaded.
8. Record revisions, visual proof, and any HF platform limitations in a short final report. Do not publish credentials or local/private data.

## 6. Completion criteria

All of the following must be checked, not assumed:

- [ ] SAEBase contains **both original PNG files**, byte-identical to the attached originals.
- [ ] Organization green card / supported brand region uses an approved logo and the wide region shows the full banner when supported; exact region verified in screenshot.
- [ ] Every Lab module has the same correct top-left symbol or banner, without broken routes or header shifts.
- [ ] Dataset card begins with the full banner below HF metadata, with all prior research information intact.
- [ ] No third OpenSAE graphic logo remains on the audited relevant pages.
- [ ] No new decorative explanations or tiny text were added.
- [ ] Desktop and mobile appearance, image loading, alt text and aspect ratios pass.
- [ ] Public links, source commits, HF revisions, screenshots and remaining blockers are reported honestly.

Produce `outputs/research_lab_brand_08/final_report.md` locally (or another existing verified report directory) containing: asset checksums, exact file changes, commit hashes, public links and before/after screenshots of all three locations.

**Stop condition:** A change to GitHub `saebase/assets/branding/` alone is NOT proof the organization card, Lab, or dataset was visually updated. The agent must update and verify each supported external surface independently.
