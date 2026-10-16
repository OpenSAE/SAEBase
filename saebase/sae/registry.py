"""SAE profiles keyed independently of base-model plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from models.gemma1b.gemma1b import SPEC as GEMMA1B_SPEC
from models.gemma1b_it.gemma1b_it import SPEC as GEMMA1B_IT_SPEC
from models.gpt2small.gpt2small import SPEC as GPT2SMALL_SPEC
from models.gemma270m.gemma270m import SPEC as GEMMA270M_SPEC
from models.gemma2_2b_base.gemma2_2b_base import SPEC as GEMMA2_2B_BASE_SPEC
from models.gemma2_2b_it.gemma2_2b_it import SPEC as GEMMA2_2B_IT_SPEC
from models.gemma2_9b_it.gemma2_9b_it import SPEC as GEMMA2_9B_IT_SPEC
from models.gemma2_9b_pt.gemma2_9b_pt import SPEC as GEMMA2_9B_PT_SPEC
from models.gemma3_4b_it.gemma3_4b_it import SPEC as GEMMA3_4B_IT_SPEC
from models.gemma3_4b_pt.gemma3_4b_pt import SPEC as GEMMA3_4B_PT_SPEC
from models.gemma3_12b_it.gemma3_12b_it import SPEC as GEMMA3_12B_IT_SPEC
from models.gemma3_12b_pt.gemma3_12b_pt import SPEC as GEMMA3_12B_PT_SPEC
from models.gemma3_27b_it.gemma3_27b_it import SPEC as GEMMA3_27B_IT_SPEC
from models.gemma3_27b_pt.gemma3_27b_pt import SPEC as GEMMA3_27B_PT_SPEC
from models.hooked_lm_common import BaseModelSpec
from models.llama31_8b.llama31_8b import SPEC as LLAMA31_8B_SPEC
from models.llama31_8b_instruct.llama31_8b_instruct import SPEC as LLAMA31_8B_INSTRUCT_SPEC
from models.pythia410m.pythia410m import SPEC as PYTHIA410M_SPEC
from models.pythia70m.pythia70m import SPEC as PYTHIA70M_SPEC
from models.qwen3_8b.qwen3_8b import SPEC as QWEN3_8B_SPEC
from models.qwen3_1_7b.qwen3_1_7b import SPEC as QWEN3_1_7B_SPEC
from models.qwen3_1_7b_base.qwen3_1_7b_base import SPEC as QWEN3_1_7B_BASE_SPEC
from models.mistral7b_instruct.mistral7b_instruct import SPEC as MISTRAL7B_INSTRUCT_SPEC
from models.qwen35_4b.qwen35_4b import SPEC as QWEN35_4B_SPEC
from models.qwen35_2b.qwen35_2b import SPEC as QWEN35_2B_SPEC
from models.r1_distill_qwen15b.r1_distill_qwen15b import SPEC as R1_DISTILL_QWEN15B_SPEC
from models.r1_distill_llama8b.r1_distill_llama8b import SPEC as R1_DISTILL_LLAMA8B_SPEC
from models.qwen35_9b.qwen35_9b import SPEC as QWEN35_9B_SPEC
from models.qwen35_27b.qwen35_27b import SPEC as QWEN35_27B_SPEC

LoaderKind = Literal["sae_lens", "eleutherai_topk_mlp", "qwen_resid_topk", "dictionary_learning", "sparsify_topk"]
TransferKind = Literal["none", "base_to_instruct"]

_BASE_SPECS: dict[str, BaseModelSpec] = {
    GPT2SMALL_SPEC.key: GPT2SMALL_SPEC,
    PYTHIA70M_SPEC.key: PYTHIA70M_SPEC,
    PYTHIA410M_SPEC.key: PYTHIA410M_SPEC,
    GEMMA270M_SPEC.key: GEMMA270M_SPEC,
    GEMMA1B_SPEC.key: GEMMA1B_SPEC,
    GEMMA1B_IT_SPEC.key: GEMMA1B_IT_SPEC,
    R1_DISTILL_QWEN15B_SPEC.key: R1_DISTILL_QWEN15B_SPEC,
    R1_DISTILL_LLAMA8B_SPEC.key: R1_DISTILL_LLAMA8B_SPEC,
    GEMMA2_2B_BASE_SPEC.key: GEMMA2_2B_BASE_SPEC,
    GEMMA2_2B_IT_SPEC.key: GEMMA2_2B_IT_SPEC,
    GEMMA3_4B_PT_SPEC.key: GEMMA3_4B_PT_SPEC,
    GEMMA3_4B_IT_SPEC.key: GEMMA3_4B_IT_SPEC,
    LLAMA31_8B_SPEC.key: LLAMA31_8B_SPEC,
    LLAMA31_8B_INSTRUCT_SPEC.key: LLAMA31_8B_INSTRUCT_SPEC,
    QWEN3_8B_SPEC.key: QWEN3_8B_SPEC,
    QWEN3_1_7B_SPEC.key: QWEN3_1_7B_SPEC,
    QWEN3_1_7B_BASE_SPEC.key: QWEN3_1_7B_BASE_SPEC,
    MISTRAL7B_INSTRUCT_SPEC.key: MISTRAL7B_INSTRUCT_SPEC,
    QWEN35_4B_SPEC.key: QWEN35_4B_SPEC,
    QWEN35_2B_SPEC.key: QWEN35_2B_SPEC,
    GEMMA2_9B_PT_SPEC.key: GEMMA2_9B_PT_SPEC,
    GEMMA2_9B_IT_SPEC.key: GEMMA2_9B_IT_SPEC,
    GEMMA3_12B_PT_SPEC.key: GEMMA3_12B_PT_SPEC,
    GEMMA3_12B_IT_SPEC.key: GEMMA3_12B_IT_SPEC,
    QWEN35_9B_SPEC.key: QWEN35_9B_SPEC,
    GEMMA3_27B_PT_SPEC.key: GEMMA3_27B_PT_SPEC,
    GEMMA3_27B_IT_SPEC.key: GEMMA3_27B_IT_SPEC,
    QWEN35_27B_SPEC.key: QWEN35_27B_SPEC,
}


@dataclass(frozen=True)
class SAEProfile:
    """One SAE release/site configuration for a base model."""

    key: str
    model_key: str
    sae_release: str
    layers: tuple[int, ...]
    hook_kind: str
    loader: LoaderKind
    sae_id_template: str | None = None
    sae_folder_template: str | None = None
    notes: str = ""
    logical_sae_id: str | None = None
    resolved_physical_path: str | None = None
    repo_id: str | None = None
    trained_on_model_key: str | None = None
    target_model_key: str | None = None
    transfer_kind: TransferKind = "none"
    hook_note: str = ""
    sae_id_by_layer: tuple[tuple[int, str], ...] = ()

    @property
    def base(self) -> BaseModelSpec:
        return _BASE_SPECS[self.model_key]

    @property
    def model_name(self) -> str:
        return self.base.model_name

    @property
    def model_candidates(self) -> tuple[str, ...]:
        return self.base.candidates()

    def allowed_model_keys(self) -> frozenset[str]:
        keys = {self.model_key}
        if self.trained_on_model_key:
            keys.add(self.trained_on_model_key)
        if self.target_model_key:
            keys.add(self.target_model_key)
        return frozenset(keys)

    def hook_name(self, layer: int) -> str:
        self._assert_layer(layer)
        return f"blocks.{layer}.{self.hook_kind}"

    def sae_id(self, layer: int) -> str:
        self._assert_layer(layer)
        mapped = dict(self.sae_id_by_layer)
        if layer in mapped:
            return mapped[layer]
        if self.logical_sae_id is not None and len(self.layers) == 1:
            return self.logical_sae_id
        if self.sae_id_template is not None:
            return self.sae_id_template.format(L=layer)
        return self.hook_name(layer)

    def sae_folder(self, layer: int) -> str | None:
        self._assert_layer(layer)
        if self.resolved_physical_path is not None and len(self.layers) == 1:
            return self.resolved_physical_path
        if self.sae_folder_template is None:
            return None
        return self.sae_folder_template.format(L=layer)

    def transfer_meta(self) -> dict[str, str | None]:
        return {
            "release": self.sae_release,
            "logical_sae_id": self.logical_sae_id or (
                self.sae_id_template.format(L=self.layers[0])
                if self.sae_id_template and self.layers
                else None
            ),
            "resolved_physical_path": self.resolved_physical_path
            or (
                self.sae_folder_template.format(L=self.layers[0])
                if self.sae_folder_template and self.layers
                else None
            ),
            "repo_id": self.repo_id,
            "trained_on_model_key": self.trained_on_model_key or self.model_key,
            "target_model_key": self.target_model_key or self.model_key,
            "transfer_kind": self.transfer_kind,
            "hook_note": self.hook_note or None,
        }

    def _assert_layer(self, layer: int) -> None:
        if layer not in self.layers:
            raise ValueError(
                f"{self.key}: layer {layer} not in selected layers {list(self.layers)}"
            )


# Back-compat alias for internal callers.
ModelProfile = SAEProfile

_PROFILES: dict[str, SAEProfile] = {
    "gpt2small_res_jb_l8": SAEProfile(
        key="gpt2small_res_jb_l8",
        model_key="gpt2small",
        sae_release="gpt2-small-res-jb",
        layers=(8,), hook_kind="hook_resid_pre", loader="sae_lens",
        logical_sae_id="blocks.8.hook_resid_pre",
        repo_id="jbloom/GPT2-Small-SAEs-Reformatted",
        notes="Original GPT-2 Small layer-8 residual SAE (24,576 latents).",
    ),
    "gpt2small_fs_6144_l8": SAEProfile(
        key="gpt2small_fs_6144_l8",
        model_key="gpt2small",
        sae_release="gpt2-small-res-jb-feature-splitting",
        layers=(8,), hook_kind="hook_resid_pre", loader="sae_lens",
        logical_sae_id="blocks.8.hook_resid_pre_6144",
        repo_id="jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8",
        notes="Same GPT-2 layer-8 site, separately trained feature-splitting SAE (6,144 latents).",
    ),
    "gpt2small_fs_12288_l8": SAEProfile(
        key="gpt2small_fs_12288_l8",
        model_key="gpt2small",
        sae_release="gpt2-small-res-jb-feature-splitting",
        layers=(8,), hook_kind="hook_resid_pre", loader="sae_lens",
        logical_sae_id="blocks.8.hook_resid_pre_12288",
        repo_id="jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8",
        notes="Optional M5 extended width, same layer-8 residual site (12,288 latents).",
    ),
    "gpt2small_fs_24576_l8": SAEProfile(
        key="gpt2small_fs_24576_l8",
        model_key="gpt2small",
        sae_release="gpt2-small-res-jb-feature-splitting",
        layers=(8,), hook_kind="hook_resid_pre", loader="sae_lens",
        logical_sae_id="blocks.8.hook_resid_pre_24576",
        repo_id="jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8",
        notes="Optional M5 extended width, same layer-8 residual site (24,576 latents).",
    ),
    "pythia70m_saebench_gated_l3": SAEProfile(
        key="pythia70m_saebench_gated_l3",
        model_key="pythia70m",
        sae_release="sae_bench_pythia70m_sweep_gated_ctx128_0730",
        layers=(3,), hook_kind="hook_resid_post", loader="sae_lens",
        logical_sae_id="blocks.3.hook_resid_post__trainer_5",
        notes="SAEBench gated SAE checkpoint validated natively in M3; same layer as res-sm.",
    ),
    "pythia70m_resid_sm": SAEProfile(
        key="pythia70m_resid_sm",
        model_key="pythia70m",
        sae_release="pythia-70m-deduped-res-sm",
        layers=(1, 3, 5),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        notes="Residual-stream SAE; sae_id == hook name.",
    ),
    "pythia410m_mlp_topk": SAEProfile(
        key="pythia410m_mlp_topk",
        model_key="pythia410m",
        sae_release="EleutherAI/sae-pythia-410m-65k",
        layers=(4, 12, 20),
        hook_kind="hook_mlp_out",
        loader="eleutherai_topk_mlp",
        sae_folder_template="layers.{L}.mlp",
        notes="EleutherAI TopK MLP SAEs (no resid release).",
    ),
    "gemma270m_scope_resid": SAEProfile(
        key="gemma270m_scope_resid",
        model_key="gemma270m",
        sae_release="gemma-scope-2-270m-pt-resid_post",
        layers=(5, 9, 12),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_template="layer_{L}_width_65k_l0_medium",
        notes="Gemma Scope 2 residual SAEs (65k, l0 medium).",
        repo_id="google/gemma-scope-2-270m-pt",
    ),
    "gemma1b_scope_resid": SAEProfile(
        key="gemma1b_scope_resid",
        model_key="gemma1b",
        sae_release="gemma-scope-2-1b-pt-resid_post",
        layers=(7, 13, 17),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_template="layer_{L}_width_16k_l0_medium",
        notes="Gemma Scope 2 residual SAEs (16k, l0 medium).",
        repo_id="google/gemma-scope-2-1b-pt",
        trained_on_model_key="gemma1b",
        target_model_key="gemma1b",
        transfer_kind="none",
    ),
    # P0 Gemma-3 1B layer-25 PT-transfer (main coordinate system).
    "gemma1b_pt_l25_transfer_big": SAEProfile(
        key="gemma1b_pt_l25_transfer_big",
        model_key="gemma1b",
        sae_release="gemma-scope-2-1b-pt-res-all",
        layers=(13, 25),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_template="layer_{L}_width_16k_l0_big",
        sae_folder_template="resid_post_all/layer_{L}_width_16k_l0_big",
        notes="P0 PT SAE; primary transfer profile for base↔IT contrasts.",
        repo_id="google/gemma-scope-2-1b-pt",
        trained_on_model_key="gemma1b",
        target_model_key="gemma1b_it",
        transfer_kind="base_to_instruct",
        hook_note="blocks.{L}.hook_resid_post (pre-final-norm)",
    ),
    "gemma1b_pt_l25_transfer_small": SAEProfile(
        key="gemma1b_pt_l25_transfer_small",
        model_key="gemma1b",
        sae_release="gemma-scope-2-1b-pt-res-all",
        layers=(25,),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_template="layer_{L}_width_16k_l0_small",
        sae_folder_template="resid_post_all/layer_{L}_width_16k_l0_small",
        notes="P0 sensitivity PT SAE (l0_small).",
        repo_id="google/gemma-scope-2-1b-pt",
        trained_on_model_key="gemma1b",
        target_model_key="gemma1b_it",
        transfer_kind="base_to_instruct",
        hook_note="blocks.25.hook_resid_post (pre-final-norm)",
    ),
    "gemma1b_it_l25_native_big": SAEProfile(
        key="gemma1b_it_l25_native_big",
        model_key="gemma1b_it",
        sae_release="gemma-scope-2-1b-it-res-all",
        layers=(25,),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_template="layer_{L}_width_16k_l0_big",
        sae_folder_template="resid_post_all/layer_{L}_width_16k_l0_big",
        notes="P0 native IT SAE for diagnosis only; not primary coordinate.",
        repo_id="google/gemma-scope-2-1b-it",
        trained_on_model_key="gemma1b_it",
        target_model_key="gemma1b_it",
        transfer_kind="none",
        hook_note="blocks.25.hook_resid_post (pre-final-norm)",
    ),
    "r1_distill_qwen15b_resid_l20": SAEProfile(
        key="r1_distill_qwen15b_resid_l20",
        model_key="r1_distill_qwen15b",
        sae_release="analist/sae-r1-distill-qwen-1.5b-resid-l20-reasoning",
        layers=(20,),
        hook_kind="hook_resid_post",
        loader="sparsify_topk",
        sae_id_template="resid_post_layer{L}",
        sae_folder_template="resid_post_layer{L}",
        notes=(
            "Reasoning-trained sparsify TopK residual SAE (65,536, k=32, d_in=1536); "
            "OpenThoughts-114k, 300M tokens, held-out reasoning FVU=0.1236. "
            "First-round ordered-division PoC."
        ),
        repo_id="analist/sae-r1-distill-qwen-1.5b-resid-l20-reasoning",
        hook_note="blocks.20.hook_resid_post (resid_post of decoder layer 20)",
    ),
    "r1_distill_llama8b_llamascope_l23": SAEProfile(
        key="r1_distill_llama8b_llamascope_l23",
        model_key="r1_distill_llama8b",
        sae_release="llama_scope_r1_distill",
        layers=(23,),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_template="l{L}r_400m_slimpajama_400m_openr1_math",
        sae_folder_template="400M-Slimpajama-400M-OpenR1-Math-220k/L{L}R",
        notes=(
            "Llama-Scope R1 mixed SlimPajama+OpenR1 residual 32k JumpReLU at L23 "
            "(~71% depth). Neuronpedia source 23-llamascope-slimpj-openr1-res-32k."
        ),
        repo_id="fnlp/Llama-Scope-R1-Distill",
        hook_note="blocks.23.hook_resid_post (post-MLP residual)",
    ),
    # P0 Gemma-2-2B canonical layer-25.
    "gemma2_2b_pt_l25_canonical": SAEProfile(
        key="gemma2_2b_pt_l25_canonical",
        model_key="gemma2_2b_base",
        sae_release="gemma-scope-2b-pt-res-canonical",
        layers=(13, 25),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_template="layer_{L}/width_16k/canonical",
        resolved_physical_path="layer_25/width_16k/average_l0_116",
        notes="P0 primary Gemma-2 canonical residual SAE (L0≈116).",
        logical_sae_id="layer_25/width_16k/canonical",
        repo_id="google/gemma-scope-2b-pt-res",
        trained_on_model_key="gemma2_2b_base",
        target_model_key="gemma2_2b_it",
        transfer_kind="base_to_instruct",
        hook_note="blocks.25.hook_resid_post (pre-final-norm)",
    ),
    "gemma3_4b_pt_scope_resid": SAEProfile(
        key="gemma3_4b_pt_scope_resid",
        model_key="gemma3_4b_pt",
        sae_release="gemma-scope-2-4b-pt-res",
        layers=(9, 17, 22, 29),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_template="layer_{L}_width_65k_l0_medium",
        sae_folder_template="resid_post/layer_{L}_width_65k_l0_medium",
        notes="Official Gemma Scope 2 25/50/65/85% subset; 65k medium.",
        repo_id="google/gemma-scope-2-4b-pt",
        hook_note="blocks.{L}.hook_resid_post (pre-final-norm)",
    ),
    "gemma3_4b_it_scope_resid": SAEProfile(
        key="gemma3_4b_it_scope_resid",
        model_key="gemma3_4b_it",
        sae_release="gemma-scope-2-4b-it-res",
        layers=(9, 17, 22, 29),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_template="layer_{L}_width_65k_l0_medium",
        sae_folder_template="resid_post/layer_{L}_width_65k_l0_medium",
        notes="Native IT Scope 2 residual; same official four depths as PT.",
        repo_id="google/gemma-scope-2-4b-it",
        hook_note="blocks.{L}.hook_resid_post (pre-final-norm)",
    ),
    "llama31_8b_lxr_8x": SAEProfile(
        key="llama31_8b_lxr_8x",
        model_key="llama31_8b",
        sae_release="OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x",
        layers=(8, 16, 21, 27),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_template="Llama3_1-8B-Base-L{L}R-8x",
        sae_folder_template="Llama3_1-8B-Base-L{L}R-8x",
        notes="Llama-Scope 8x residual (32k, TopK≈50) at ~25/50/65/85%.",
        repo_id="OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x",
        hook_note="blocks.{L}.hook_resid_post (post-MLP residual)",
    ),
    "qwen35_2b_resid_w32k": SAEProfile(
        key="qwen35_2b_resid_w32k",
        model_key="qwen35_2b",
        sae_release="Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50",
        layers=(6, 12, 15, 20),
        hook_kind="hook_resid_post",
        loader="qwen_resid_topk",
        sae_id_template="layer{L}.sae.pt",
        notes="Official Qwen Scope 3.5 2B, 32k TopK-50 residual SAEs; pending end-to-end real-model verification.",
        repo_id="Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50",
        hook_note="blocks.{L}.hook_resid_post",
    ),
    "qwen3_8b_resid_w64k": SAEProfile(
        key="qwen3_8b_resid_w64k",
        model_key="qwen3_8b",
        sae_release="Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
        layers=(9, 18, 23, 31),
        hook_kind="hook_resid_post",
        loader="qwen_resid_topk",
        sae_id_template="layer{L}.sae.pt",
        notes="Official Qwen residual TopK 64k / K=50; 36-layer model.",
        repo_id="Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
        hook_note="blocks.{L}.hook_resid_post",
    ),
    "gemma2_9b_pt_scope_resid": SAEProfile(
        key="gemma2_9b_pt_scope_resid",
        model_key="gemma2_9b_pt",
        sae_release="gemma-scope-9b-pt-res",
        layers=(9, 20, 31),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_by_layer=(
            (9, "layer_9/width_16k/average_l0_51"),
            (20, "layer_20/width_16k/average_l0_58"),
            (31, "layer_31/width_16k/average_l0_35"),
        ),
        notes="Gemma Scope 16k; 3 layers aligned with native IT SAE sites.",
        repo_id="google/gemma-scope-9b-pt-res",
        hook_note="blocks.{L}.hook_resid_post (pre-final-norm)",
    ),
    "gemma2_9b_it_scope_resid": SAEProfile(
        key="gemma2_9b_it_scope_resid",
        model_key="gemma2_9b_it",
        sae_release="gemma-scope-9b-it-res",
        layers=(9, 20, 31),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_by_layer=(
            (9, "layer_9/width_16k/average_l0_47"),
            (20, "layer_20/width_16k/average_l0_91"),
            (31, "layer_31/width_16k/average_l0_43"),
        ),
        notes="Native IT Scope residual; L20 uses AxBench average_l0_91 (not l0_47).",
        repo_id="google/gemma-scope-9b-it-res",
        hook_note="blocks.{L}.hook_resid_post (pre-final-norm)",
    ),
    "gemma3_12b_pt_scope_resid": SAEProfile(
        key="gemma3_12b_pt_scope_resid",
        model_key="gemma3_12b_pt",
        sae_release="gemma-scope-2-12b-pt-res",
        layers=(12, 24, 31, 41),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_template="layer_{L}_width_65k_l0_medium",
        sae_folder_template="resid_post/layer_{L}_width_65k_l0_medium",
        notes="Official Gemma Scope 2 four-depth subset; 65k medium.",
        repo_id="google/gemma-scope-2-12b-pt",
        hook_note="blocks.{L}.hook_resid_post (pre-final-norm)",
    ),
    "gemma3_12b_it_scope_resid": SAEProfile(
        key="gemma3_12b_it_scope_resid",
        model_key="gemma3_12b_it",
        sae_release="gemma-scope-2-12b-it-res",
        layers=(12, 24, 31, 41),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_template="layer_{L}_width_65k_l0_medium",
        sae_folder_template="resid_post/layer_{L}_width_65k_l0_medium",
        notes="Native IT Scope 2 residual; same official four depths as PT.",
        repo_id="google/gemma-scope-2-12b-it",
        hook_note="blocks.{L}.hook_resid_post (pre-final-norm)",
    ),
    "qwen35_9b_resid_w64k": SAEProfile(
        key="qwen35_9b_resid_w64k",
        model_key="qwen35_9b",
        sae_release="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50",
        layers=(8, 16, 21, 27),
        hook_kind="hook_resid_post",
        loader="qwen_resid_topk",
        sae_id_template="layer{L}.sae.pt",
        notes="Official Qwen3.5 residual TopK 64k / K=50; 32-layer model.",
        repo_id="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50",
        hook_note="blocks.{L}.hook_resid_post",
    ),
    "gemma3_27b_pt_scope_resid": SAEProfile(
        key="gemma3_27b_pt_scope_resid",
        model_key="gemma3_27b_pt",
        sae_release="gemma-scope-2-27b-pt-res",
        layers=(16, 31, 40, 53),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_template="layer_{L}_width_65k_l0_medium",
        sae_folder_template="resid_post/layer_{L}_width_65k_l0_medium",
        notes="Official Gemma Scope 2 four-depth subset; 65k medium.",
        repo_id="google/gemma-scope-2-27b-pt",
        hook_note="blocks.{L}.hook_resid_post (pre-final-norm)",
    ),
    "gemma3_27b_it_scope_resid": SAEProfile(
        key="gemma3_27b_it_scope_resid",
        model_key="gemma3_27b_it",
        sae_release="gemma-scope-2-27b-it-res",
        layers=(16, 31, 40, 53),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_template="layer_{L}_width_65k_l0_medium",
        sae_folder_template="resid_post/layer_{L}_width_65k_l0_medium",
        notes="Native IT Scope 2 residual; same official four depths as PT.",
        repo_id="google/gemma-scope-2-27b-it",
        hook_note="blocks.{L}.hook_resid_post (pre-final-norm)",
    ),
    "qwen35_27b_resid_w80k": SAEProfile(
        key="qwen35_27b_resid_w80k",
        model_key="qwen35_27b",
        sae_release="Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50",
        layers=(16, 32, 42, 54),
        hook_kind="hook_resid_post",
        loader="qwen_resid_topk",
        sae_id_template="layer{L}.sae.pt",
        notes="Official Qwen3.5 27B residual TopK 80k / K=50; 64-layer model.",
        repo_id="Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50",
        hook_note="blocks.{L}.hook_resid_post",
    ),
    "llama31_8b_instruct_andyrdt_l19_k64": SAEProfile(
        key="llama31_8b_instruct_andyrdt_l19_k64",
        model_key="llama31_8b_instruct",
        sae_release="andyrdt/saes-llama-3.1-8b-instruct",
        layers=(19,),
        hook_kind="hook_resid_post",
        loader="dictionary_learning",
        sae_id_template="resid_post_layer_{L}/trainer_1",
        sae_folder_template="resid_post_layer_{L}/trainer_1",
        notes="BatchTopK k=64; inference uses checkpoint learned threshold, not TopK-64.",
        repo_id="andyrdt/saes-llama-3.1-8b-instruct",
        hook_note="blocks.{L}.hook_resid_post",
    ),
    # Protocol 11a upgrades (do not change DEFAULT_SAE_FOR_MODEL).
    "gemma2_9b_it_scope_resid_l31_131k": SAEProfile(
        key="gemma2_9b_it_scope_resid_l31_131k",
        model_key="gemma2_9b_it",
        sae_release="gemma-scope-9b-it-res",
        layers=(31,),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_by_layer=((31, "layer_31/width_131k/average_l0_109"),),
        notes="Experiment 11a: GemmaScope IT residual L31 / 131k / average_l0_109.",
        repo_id="google/gemma-scope-9b-it-res",
        hook_note="blocks.{L}.hook_resid_post (pre-final-norm)",
    ),
    "llama31_8b_instruct_andyrdt_l23_k64": SAEProfile(
        key="llama31_8b_instruct_andyrdt_l23_k64",
        model_key="llama31_8b_instruct",
        sae_release="andyrdt/saes-llama-3.1-8b-instruct",
        layers=(23,),
        hook_kind="hook_resid_post",
        loader="dictionary_learning",
        sae_id_template="resid_post_layer_{L}/trainer_1",
        sae_folder_template="resid_post_layer_{L}/trainer_1",
        notes="Experiment 11a: BatchTopK k=64 at L23 trainer_1 (Neuronpedia 23-resid-post-aa).",
        repo_id="andyrdt/saes-llama-3.1-8b-instruct",
        hook_note="blocks.{L}.hook_resid_post",
    ),
    # Experiment 11b (do not change DEFAULT_SAE_FOR_MODEL).
    "gemma2_2b_it_scope_resid_l12": SAEProfile(
        key="gemma2_2b_it_scope_resid_l12",
        model_key="gemma2_2b_base",
        sae_release="gemma-scope-2b-pt-res",
        layers=(12,),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_template="layer_{L}/width_16k/average_l0_82",
        sae_folder_template="layer_{L}/width_16k/average_l0_82",
        notes=(
            "Experiment 11b: GemmaScope 2B PT residual L12 / 16k / average_l0_82 "
            "(~46% depth; PT→IT transfer onto gemma2_2b_it)."
        ),
        repo_id="google/gemma-scope-2b-pt-res",
        trained_on_model_key="gemma2_2b_base",
        target_model_key="gemma2_2b_it",
        transfer_kind="base_to_instruct",
        hook_note="blocks.12.hook_resid_post (pre-final-norm)",
    ),
    "qwen3_1_7b_base_resid_w32k": SAEProfile(
        key="qwen3_1_7b_base_resid_w32k",
        model_key="qwen3_1_7b_base",
        sae_release="Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50",
        layers=(7, 14, 18, 23),
        hook_kind="hook_resid_post",
        loader="qwen_resid_topk",
        sae_id_template="layer{L}.sae.pt",
        repo_id="Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50",
        notes="Official Qwen Scope Base TopK-50 residual (32k); distinct from Qwen3-1.7B chat.",
    ),
    "mistral7b_instruct_tc_l16_131k": SAEProfile(
        key="mistral7b_instruct_tc_l16_131k",
        model_key="mistral7b_instruct",
        sae_release="tylercosgrove/mistral-7b-sparse-autoencoder-layer16",
        layers=(16,),
        hook_kind="hook_mlp_out",
        loader="sae_lens",
        logical_sae_id=".",
        repo_id="tylercosgrove/mistral-7b-sparse-autoencoder-layer16",
        notes=(
            "Community TopK-128 131072-latent layer-16 SAE (~4.3 GB). "
            "Published cfg.json hook_name is blocks.16.hook_mlp_out "
            "(catalog previously listed resid_pre; checkpoint metadata wins)."
        ),
        hook_note="blocks.16.hook_mlp_out per tylercosgrove cfg.json",
    ),
    "qwen3_1_7b_ak_resid_l14_65k": SAEProfile(
        key="qwen3_1_7b_ak_resid_l14_65k",
        model_key="qwen3_1_7b",
        sae_release="adamkarvonen/qwen3-1.7b-saes",
        layers=(14,),
        hook_kind="hook_resid_post",
        loader="dictionary_learning",
        sae_id_template=(
            "saes_Qwen_Qwen3-1.7B_batch_top_k/resid_post_layer_{L}/trainer_2"
        ),
        sae_folder_template=(
            "saes_Qwen_Qwen3-1.7B_batch_top_k/resid_post_layer_{L}/trainer_2"
        ),
        notes=(
            "Experiment 11b: adamkarvonen BatchTopK 65k k=80 at L14 trainer_2 "
            "(Neuronpedia 14-resid-batchtopk-65k__l0-80; eval L0≈89.5)."
        ),
        repo_id="adamkarvonen/qwen3-1.7b-saes",
        hook_note="blocks.14.hook_resid_post",
    ),
    # Experiment 11d extra profiles (do not change other models' defaults).
    "gemma2_2b_it_matryoshka_l12_32k": SAEProfile(
        key="gemma2_2b_it_matryoshka_l12_32k",
        model_key="gemma2_2b_base",
        sae_release="gemma-2-2b-res-matryoshka-dc",
        layers=(12,),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_template="blocks.{L}.hook_resid_post",
        sae_folder_template="standard/blocks.{L}.hook_resid_post",
        notes=(
            "Experiment 11d: chanind Matryoshka BatchTopK 32k training-k=40 at L12 "
            "(Neuronpedia gemma-2-2b/12-res-matryoshka-dc; PT→IT onto gemma2_2b_it). "
            "Inference uses calibrated global threshold, not batch TopK."
        ),
        repo_id="chanind/gemma-2-2b-batch-topk-matryoshka-saes-w-32k-l0-40",
        trained_on_model_key="gemma2_2b_base",
        target_model_key="gemma2_2b_it",
        transfer_kind="base_to_instruct",
        hook_note="blocks.12.hook_resid_post (pre-final-norm)",
    ),
    "r1_distill_llama8b_llamascope_l15_openr1": SAEProfile(
        key="r1_distill_llama8b_llamascope_l15_openr1",
        model_key="r1_distill_llama8b",
        sae_release="llama_scope_r1_distill",
        layers=(15,),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        sae_id_template="l{L}r_800m_openr1_math",
        sae_folder_template="0-Slimpajama-800M-OpenR1-Math-220k/L{L}R",
        notes=(
            "Experiment 11d: Llama-Scope R1 OpenR1 residual 32k JumpReLU at L15 "
            "(~47% depth; Neuronpedia 15-llamascope-openr1-res-32k). "
            "Not the 11b L23 mixed SlimPajama+OpenR1 SAE."
        ),
        repo_id="fnlp/Llama-Scope-R1-Distill",
        hook_note="blocks.15.hook_resid_post (post-MLP residual)",
    ),
    "qwen35_4b_matryoshka_l15_65k": SAEProfile(
        key="qwen35_4b_matryoshka_l15_65k",
        model_key="qwen35_4b",
        sae_release="decoderesearch/qwen-3.5-saes",
        layers=(15,),
        hook_kind="hook_resid_post",
        loader="sae_lens",
        logical_sae_id="qwen-3.5-4b/btk-mat-layer-15-k-100",
        sae_folder_template="qwen-3.5-4b/btk-mat-layer-15-k-100",
        notes=(
            "Experiment 11d: decoderesearch Matryoshka BatchTopK 65k training-k=100 "
            "at L15 on Qwen3.5-4B (Neuronpedia qwen3.5-4b/15-res-matryoshka-65k). "
            "Inference uses calibrated global threshold."
        ),
        repo_id="decoderesearch/qwen-3.5-saes",
        hook_note="blocks.15.hook_resid_post",
    ),
}

# Default SAE profile for each base model (IT models use transfer/native explicitly).
DEFAULT_SAE_FOR_MODEL: dict[str, str] = {
    "gpt2small": "gpt2small_res_jb_l8",
    "pythia70m": "pythia70m_resid_sm",
    "pythia410m": "pythia410m_mlp_topk",
    "gemma270m": "gemma270m_scope_resid",
    "gemma1b": "gemma1b_scope_resid",
    "gemma1b_it": "gemma1b_it_l25_native_big",
    "r1_distill_qwen15b": "r1_distill_qwen15b_resid_l20",
    "r1_distill_llama8b": "r1_distill_llama8b_llamascope_l23",
    "gemma2_2b_base": "gemma2_2b_pt_l25_canonical",
    "gemma2_2b_it": "gemma2_2b_pt_l25_canonical",
    "gemma3_4b_pt": "gemma3_4b_pt_scope_resid",
    "gemma3_4b_it": "gemma3_4b_it_scope_resid",
    "llama31_8b": "llama31_8b_lxr_8x",
    "llama31_8b_instruct": "llama31_8b_instruct_andyrdt_l19_k64",
    "qwen3_8b": "qwen3_8b_resid_w64k",
    "qwen35_2b": "qwen35_2b_resid_w32k",
    "qwen3_1_7b": "qwen3_1_7b_ak_resid_l14_65k",
    "qwen3_1_7b_base": "qwen3_1_7b_base_resid_w32k",
    "mistral7b_instruct": "mistral7b_instruct_tc_l16_131k",
    "qwen35_4b": "qwen35_4b_matryoshka_l15_65k",
    "gemma2_9b_pt": "gemma2_9b_pt_scope_resid",
    "gemma2_9b_it": "gemma2_9b_it_scope_resid",
    "gemma3_12b_pt": "gemma3_12b_pt_scope_resid",
    "gemma3_12b_it": "gemma3_12b_it_scope_resid",
    "qwen35_9b": "qwen35_9b_resid_w64k",
    "gemma3_27b_pt": "gemma3_27b_pt_scope_resid",
    "gemma3_27b_it": "gemma3_27b_it_scope_resid",
    "qwen35_27b": "qwen35_27b_resid_w80k",
}

SAE_KEYS: tuple[str, ...] = tuple(_PROFILES.keys())
MODEL_KEYS: tuple[str, ...] = tuple(DEFAULT_SAE_FOR_MODEL.keys())


def get_profile(sae_key: str) -> SAEProfile:
    key = str(sae_key)
    # Accept bare model_key as alias to default SAE profile.
    if key in DEFAULT_SAE_FOR_MODEL:
        key = DEFAULT_SAE_FOR_MODEL[key]
    if key not in _PROFILES:
        raise KeyError(f"Unknown sae.key={sae_key!r}. Valid: {list(SAE_KEYS)}")
    return _PROFILES[key]


def get_profile_for_model(model_key: str) -> SAEProfile:
    if model_key not in DEFAULT_SAE_FOR_MODEL:
        raise KeyError(f"Unknown model_key={model_key!r}")
    return get_profile(DEFAULT_SAE_FOR_MODEL[model_key])


def list_profiles() -> list[SAEProfile]:
    return [_PROFILES[k] for k in SAE_KEYS]


def final_layer(n_layers: int) -> int:
    return int(n_layers) - 1


def reference_layer(n_layers: int) -> int:
    import math

    return int(math.ceil(int(n_layers) / 2))
