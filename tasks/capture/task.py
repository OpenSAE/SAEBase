"""SAEBase live capture task."""

from tasks.capture.configuration import capture_section, migrate_capture_task
from saebase.runtime.sae_operations import SAEOperationTask


class CaptureTask(SAEOperationTask):
    OPERATION = "capture"

    def continuation_configuration(self, configuration):
        from tasks.capture.selection import active_whiten

        value = dict(configuration)
        task = migrate_capture_task(value["task"])
        sections = task.setdefault("capture", {})
        sections.setdefault("output_scope", "last_token")
        if sections["output_scope"] == "all_tokens":
            sections.setdefault("max_tokens", 128)
            sections.setdefault("max_active_per_token", 256)
        else:
            sections.pop("max_tokens", None)
            sections.pop("max_active_per_token", None)
        mode = sections.setdefault("mode", "non_whiten")
        sections.setdefault("whiten_chunk_size", 256)
        filtered = type(capture_section(task, "selection").get("candidate_count")) is int
        for section, active in {
            "output": mode != "rss",
            "answer": mode in {"sar", "g_s", "I"},
            "sar": mode == "sar",
            "i": mode == "I",
            "g_s": mode == "g_s" and not active_whiten(task),
            "rss": mode == "rss",
        }.items():
            if not active:
                sections.pop(section, None)
        if active_whiten(task):
            sections["selection"].pop("feature_selection", None)
            if mode == "I" and "i" in sections:
                sections["i"].pop("feature_selection", None)
                sections["i"].pop("g_s_shortfall", None)
        if mode not in {"sar", "g_s", "I"} and not filtered and "selection" in sections:
            selection = sections["selection"]
            if "candidate_count" in selection:
                sections["selection"] = {"candidate_count": selection["candidate_count"]}
            else:
                sections.pop("selection")
        from tasks.capture.model_answer import settings as answer_settings

        sections["model_answer"] = answer_settings({"task": task})
        if (
            not filtered
            and mode not in {"whiten", "g_s", "I"}
            and not (
                mode == "sar"
                and capture_section(task, "selection").get("feature_selection") == "whiten"
            )
        ):
            sections.pop("whiten_chunk_size", None)
        return {**value, "task": task}

    def continuation_action(self, result):
        summary = result.summary
        answer_stage = summary.get("stages", {}).get("model_answer")
        if answer_stage is not None and answer_stage.get("status") != "complete":
            return "retry"
        if (
            summary.get("mode") == "sar"
            and summary.get("sar", {}).get("identity", {}).get("algorithm") != "exhaustive"
            and summary.get("input_identity", {}).get("execution", {}).get("sar_algorithm")
            != "exhaustive"
        ):
            from pathlib import Path
            import hashlib

            source = summary.get("sar", {}).get("identity") or summary.get(
                "input_identity", {}
            ).get("sar", {})
            try:
                digest = hashlib.sha256(Path(source["source"]).read_bytes()).hexdigest()
            except (KeyError, OSError):
                return "retry"
            if digest != source.get("source_sha256"):
                return "retry"
        if summary.get("capture_format_version") != 3:
            return super().continuation_action(result)
        if result.status != "executed" or summary.get("source_error"):
            return "retry"
        stage = summary.get("stages", {}).get("capture")
        if stage is not None:
            expected = summary.get(
                "planned_count",
                summary.get("input_identity", {}).get("selection", {}).get("selected_count"),
            )
            captured = (
                type(expected) is int
                and expected > 0
                and stage.get("status") == "complete"
                and stage.get("evaluated", stage.get("completed")) == expected
                and summary.get("selection_exhausted") is True
                and summary.get("sample_count") == summary.get("ledger_rows") == expected
            )
        else:
            # Read historical batch records without rewriting or upgrading them.
            counts = result.completion.get("counts", {})
            expected = (
                summary.get("input_identity", {})
                .get("selection", {})
                .get("selected_count", summary.get("selection", {}).get("selected_count"))
            )
            captured = (
                type(expected) is int
                and expected > 0
                and summary.get("selection_exhausted") is True
                and summary.get("sample_count") == summary.get("ledger_rows") == expected
                and counts.get("completed", 0) + counts.get("blocked", 0) == expected
                and not any(counts.get(key, 0) for key in ("failed", "unsupported", "undefined"))
            )
        if not captured:
            return "retry"
        return "skip" if result.exit_code == 0 else "annotations_pending"

    def required_result_artifacts(self, result):
        declared = super().required_result_artifacts(result)
        if declared or result.summary.get("capture_format_version") != 3:
            return declared
        # Older batches did not record a byte count. Require their ledger to exist.
        return [{"path": result.summary["samples_path"]}]

    def read_result(self, folder, manifest):
        from tasks.capture.annotations import read_result

        from tasks.capture.model_answer import project

        updated = read_result(folder, manifest)
        return project(folder, updated)

    def refresh_annotations(self, folder):
        from tasks.capture.annotations import refresh_annotations

        return refresh_annotations(folder)

    def read_samples(self, folder):
        from tasks.capture.annotations import read_samples

        return read_samples(folder)

    def export_report(self, folder):
        from tasks.capture.report import export_report

        return export_report(folder)

    def refresh_model_answers(self, folder):
        """Complete only original model answers, preserving recorded features."""
        from tasks.capture.model_answer import refresh

        return refresh(folder)

    def run_validated(self, cfg):
        from pathlib import Path
        import json
        from saebase.results import EvaluationResult

        root = Path.cwd()
        manifest_path = root / "metadata/manifest.json"
        if cfg.get("runtime", {}).get("mode") == "resume" and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            result = manifest.get("result") or {}
            stages = result.get("summary", {}).get("stages", {})
            stage = stages.get("capture") or stages.get("features_and_annotations", {})
            if stage.get("status") == "complete":
                self.refresh_model_answers(root)
                updated = self.read_result(root, manifest)
                return EvaluationResult(**updated["result"])
        return super().run_validated(cfg)
