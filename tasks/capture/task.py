"""SAEBase live capture task."""

from tasks.sae_runtime import SAEOperationTask


class CaptureTask(SAEOperationTask):
    OPERATION = "capture"

    def continuation_configuration(self, configuration):
        value = dict(configuration)
        task = dict(value["task"])
        task.setdefault("output_scope", "last_token")
        if task["output_scope"] == "all_tokens":
            task.setdefault("max_tokens", 128)
            task.setdefault("max_active_per_token", 256)
        else:
            task.pop("max_tokens", None)
            task.pop("max_active_per_token", None)
        task.setdefault("mode", "non_whiten")
        task.setdefault("whiten_chunk_size", 256)
        filtered = type(task.get("filter")) is int
        if task["mode"] not in {"sar", "g_s", "I"} and not filtered:
            task.pop("sar", None)
        if (
            not filtered
            and task["mode"] not in {"whiten", "g_s", "I"}
            and not (
                task["mode"] == "sar" and task.get("sar", {}).get("feature_selection") == "whiten"
            )
        ):
            task.pop("whiten_chunk_size", None)
        return {**value, "task": task}

    def continuation_action(self, result):
        summary = result.summary
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
                and stage.get("evaluated" if summary.get("mode") == "sar" else "completed")
                == expected
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

        return read_result(folder, manifest)

    def refresh_annotations(self, folder):
        from tasks.capture.annotations import refresh_annotations

        return refresh_annotations(folder)

    def read_samples(self, folder):
        from tasks.capture.annotations import read_samples

        return read_samples(folder)

    def export_report(self, folder):
        from tasks.capture.report import export_report

        return export_report(folder)
