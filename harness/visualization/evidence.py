"""Connect existing scientific and project concepts to exact Python implementations.

Bindings refer only to AST-authenticated Code nodes. Every excerpt is generated
from tracked source and includes a complete callable interface for inspection.
"""
from __future__ import annotations

import copy
import ast
from harness.code.interfaces import parse_file, function_interface

CODE_TARGETS = {
    "q.modularity": "code:function:dlbase.registry:register_dataset",
    "s.registry": "code:function:dlbase.registry:register_dataset",
    "s.compatibility": "code:function:research_datasets.vision.CUB_200.dataset_task_policy:assert_task_supported",
    "s.hydra": "code:function:dlbase.train:main",
    "g.registry": "code:function:dlbase.registry:register_model",
    "g.dataset_gate": "code:function:research_datasets.vision.CUB_200.dataset_task_policy:assert_task_supported",
    "g.model_gate": "code:function:models.vision.resnet18_mlp.model_task_dataset_compatibility:assert_compatible",
    "g.pipeline": "code:function:tasks.vision.classification.pipeline_checks:run_final_pre_training_checks",
    "g.config": "code:function:dlbase.train:main",
    "g.artifact_schema": "code:function:dlbase.experiment_paths:prepare_run",
    "g.cub_dataset": "code:class:research_datasets.vision.CUB_200.CUB_200:CUB200",
    "g.cub_model": "code:class:models.vision.resnet18_mlp.resnet18_mlp:Resnet18MLP",
    "g.cub_optimization": "code:method:tasks.vision.classification.classification_module:ClassificationModule.configure_optimizers",
    "q.cub.reproducibility": "code:function:dlbase.experiment_paths:prepare_run",
    "s.cub.data": "code:function:research_datasets.vision.CUB_200.datamodule_setup:setup_cub_datamodule",
    "s.cub.model": "code:method:models.vision.resnet18_mlp.resnet18_mlp:Resnet18MLP.forward",
    "s.cub.training": "code:method:tasks.vision.classification.classification_module:ClassificationModule.training_step",
    "s.cub.provenance": "code:function:dlbase.experiment_paths:prepare_run",
    "g.cub.dataset": "code:class:research_datasets.vision.CUB_200.CUB_200:CUB200",
    "g.cub.transforms": "code:function:research_datasets.vision.CUB_200.datamodule_setup:setup_cub_datamodule",
    "g.cub.resnet": "code:method:models.vision.resnet18_mlp.resnet18_mlp:Resnet18MLP.forward",
    "g.cub.logits": "code:method:models.vision.resnet18_mlp.resnet18_mlp:Resnet18MLP.forward",
    "g.cub.loss": "code:method:tasks.vision.classification.classification_module:ClassificationModule.training_step",
    "g.cub.optimizer": "code:method:tasks.vision.classification.classification_module:ClassificationModule.configure_optimizers",
    "g.cub.schedule": "code:function:dlbase.training.schedulers:build_warmup_cosine",
    "g.cub.config": "code:function:dlbase.train:main",
    "g.cub.seed": "code:function:dlbase.training.seed:apply_lightning_reproducibility",
    "g.cub.artifacts": "code:function:dlbase.experiment_paths:prepare_run",
}
TERM_TARGETS = {
    "term.cub200": "code:class:research_datasets.vision.CUB_200.CUB_200:CUB200",
    "term.fine_grained": "code:class:models.vision.resnet18_mlp.resnet18_mlp:Resnet18MLP",
    "term.official_split": "code:function:research_datasets.vision.CUB_200.datamodule_setup:setup_cub_datamodule",
    "term.validation_set": "code:function:research_datasets.vision.CUB_200.datamodule_setup:setup_cub_datamodule",
    "term.independent_test": "code:function:tasks.vision.classification.post_train_eval:evaluate_classification_loader",
    "term.data_leakage": "code:function:research_datasets.vision.CUB_200.datamodule_setup:cub_eval_split_note",
    "term.heldout_fraction": "code:function:research_datasets.vision.CUB_200.datamodule_setup:setup_cub_datamodule",
    "term.heldout_ratio": "code:function:research_datasets.vision.CUB_200.datamodule_setup:setup_cub_datamodule",
    "term.augmentation": "code:function:research_datasets.vision.CUB_200.datamodule_setup:setup_cub_datamodule",
    "term.preprocessing": "code:function:research_datasets.vision.CUB_200.datamodule_setup:setup_cub_datamodule",
    "term.input_resolution": "code:function:research_datasets.vision.CUB_200.datamodule_setup:setup_cub_datamodule",
    "term.resnet18": "code:method:models.vision.resnet18_mlp.resnet18_mlp:Resnet18MLP.forward",
    "term.residual_connection": "code:method:models.vision.resnet18_mlp.resnet18_mlp:Resnet18MLP.forward",
    "term.backbone": "code:class:models.vision.resnet18_mlp.resnet18_mlp:Resnet18MLP",
    "term.classification_head": "code:method:models.vision.resnet18_mlp.resnet18_mlp:Resnet18MLP.forward",
    "term.logits": "code:method:tasks.vision.classification.classification_module:ClassificationModule.forward",
    "term.generalization": "code:function:tasks.vision.classification.post_train_eval:evaluate_classification_loader",
    "term.cross_entropy": "code:method:tasks.vision.classification.classification_module:ClassificationModule.training_step",
    "term.label_smoothing": "code:method:tasks.vision.classification.classification_module:ClassificationModule.training_step",
    "term.sgd": "code:method:tasks.vision.classification.classification_module:ClassificationModule.configure_optimizers",
    "term.momentum": "code:method:tasks.vision.classification.classification_module:ClassificationModule.configure_optimizers",
    "term.weight_decay": "code:method:tasks.vision.classification.classification_module:ClassificationModule.configure_optimizers",
    "term.batch_size": "code:function:research_datasets.vision.CUB_200.datamodule_setup:setup_cub_datamodule",
    "term.learning_rate": "code:method:tasks.vision.classification.classification_module:ClassificationModule.configure_optimizers",
    "term.cosine": "code:function:dlbase.training.schedulers:build_warmup_cosine",
    "term.epoch": "code:method:tasks.vision.classification.classification_module:ClassificationModule.training_step",
    "term.checkpoint": "code:function:dlbase.training.standard_lightning:build_model_checkpoint_callbacks",
    "term.model_selection": "code:function:dlbase.training.standard_lightning:build_model_checkpoint_callbacks",
    "term.top1": "code:function:tasks.vision.classification.post_train_eval:evaluate_classification_loader",
    "term.top5": "code:function:tasks.vision.classification.post_train_eval:evaluate_classification_loader",
    "term.hydra": "code:function:dlbase.train:main",
    "term.seed": "code:function:dlbase.training.seed:apply_lightning_reproducibility",
    "term.reproducibility": "code:function:dlbase.experiment_paths:prepare_run",
    "term.lightning": "code:class:tasks.vision.classification.classification_module:ClassificationModule",
}



# Core infrastructure code sits in Harness and is imported directly from real
# source. It is a separate module from the production Code Tree's default scope.
DIRECT_TARGETS = {
    "q.transparency": ("harness/core/validate.py", "main", "校验项目架构节点与父子结构。"),
    "s.explainability": ("harness/visualization/export.py", "main", "生成用于浏览的结构化项目视图。"),
    "g.context": ("harness/knowledge/validate.py", "validate_scientific", "校验科研知识、术语字典与关联结构。"),
    "g.code_tree": ("harness/code/build.py", "build", "从实际 Python AST 生成代码树。"),
    "g.env_tree": ("harness/environment/probe.py", "profile", "采集当前机器的硬件与执行信息。"),
    "g.transfer": ("harness/transfer/b2.py", "command", "构建现有云端复制与校验命令。"),
}


def direct_source_attribute(key: str) -> dict:
    path, target, explanation = DIRECT_TARGETS[key]
    module,text=parse_file(path)
    fn=next((n for n in module.body
             if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==target),None)
    if fn is None:
        raise ValueError(f"Source function for {key} is absent: {path}::{target}")
    info=function_interface(fn,text,summary=explanation)
    return {
        "type":"code_reference","label":"对应的真实代码","title":explanation,
        "symbol_id":None,"path":path,"line":info["line"],"interface":info,
    }


def real_code_interface(symbol: dict) -> dict:
    """Present one original signature, its IO and a truthful source excerpt."""
    kind = symbol.get("kind")
    blocks=symbol.get("attributes",[])
    info=next((b["interface"] for b in blocks if b.get("type")=="interface"),None)
    if kind=="class":
        container=next((b for b in blocks if b.get("type")=="class_methods"),None)
        methods=container.get("methods",[]) if container else []
        preferred=next((m for m in methods if m["name"] in ("forward","run","__getitem__","setup")),None)
        if preferred is None and methods:
            preferred=methods[0]
        info=preferred
    if kind=="module":
        excerpt=next((b for b in blocks if b.get("type")=="code"),None)
        if excerpt:
            return {"type":"code_reference","label":"对应的真实代码",
                    "title":symbol["title"],"symbol_id":symbol["id"],
                    "path":symbol["path"],"code":excerpt["value"],
                    "source_type":"module"}
    if info is None:
        return {"type":"reference","label":"对应的代码实体",
                "path":symbol.get("path"),"line":symbol.get("line")}
    return {
        "type":"code_reference","label":"对应的真实代码",
        "title":symbol.get("title"),
        "symbol_id":symbol["id"],
        "path":symbol.get("path"),
        "line":info.get("line",symbol.get("line")),
        "interface":copy.deepcopy(info),
    }



def sae_method_evidence(node: dict, code_symbols: dict) -> dict | None:
    """Attach an actual Code object for a source-backed Knowledge SAE method.

    Prefer a real public function in the cited module. If a module only exposes
    private implementation helpers, the source-backed module reference remains
    explicit without inventing a public callable signature.
    """
    paths = [str(p).split(":", 1)[0] for p in node.get("evidence", [])]
    for path in paths:
        candidates = [symbol for symbol in code_symbols.values()
                      if symbol.get("path") == path]
        if not candidates:
            continue
        callables = sorted((symbol for symbol in candidates
                            if symbol.get("kind") in {"function", "class", "method"}),
                           key=lambda item: (
                               0 if item.get("source_symbol") == "execute" else 1,
                               0 if item.get("kind") == "function" else 1,
                               int(item.get("line") or 0), item["id"]))
        for symbol in callables:
            detail = real_code_interface(symbol)
            if detail.get("type") == "code_reference":
                return detail
        modules = [symbol for symbol in candidates
                   if symbol.get("kind") == "module"]
        if modules:
            module = modules[0]
            return {"type": "code_reference", "label": "对应的真实代码",
                    "title": module["title"], "symbol_id": module["id"],
                    "path": path, "line": module.get("line", 1),
                    "source_type": "module"}
    return None


def attach_evidence(tree: dict, code: dict, mapping: dict[str,str], *,
                    skip_ids: set[str] | None=None) -> None:
    symbols={n["id"]:n for n in code["nodes"]}
    visible={n["id"] for n in tree["nodes"]}
    allowed=set(mapping)-set(skip_ids or ())
    for node in tree["nodes"]:
        key=node["id"]
        if key in DIRECT_TARGETS:
            node.setdefault("attributes",[]).append(direct_source_attribute(key))
            continue
        if key not in allowed:
            if key.startswith(("s.sae.", "g.sae.")):
                detail = sae_method_evidence(node, symbols)
                if detail is None:
                    raise ValueError("Missing real SAE source evidence for " + key)
                node.setdefault("attributes", []).append(detail)
            continue
        symbol_id=mapping[key]
        symbol=symbols.get(symbol_id)
        if symbol is None:
            raise ValueError(f"Evidence binding for {key} references an absent code entity: {symbol_id}")
        detail=real_code_interface(symbol)
        attrs=node.setdefault("attributes",[])
        attrs.append(detail)
        if symbol_id in visible:
            node.setdefault("reference_ids",[]).append(symbol_id)
