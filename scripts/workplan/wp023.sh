#!/usr/bin/env bash
# Source-owned, independent WP23 local-agent handoff.
# All real model/data/SAE checks remain on the operator's machine.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  echo "Usage: bash scripts/workplan/wp023.sh <setup|inspect|inventory|asset_prepare|dataset|model|sae|compatibility|task|audit|handoff|run|resume> [--run-id ID] [--allow-compute] [--allow-gpu] [--allow-network] [--allow-install] [--allow-download] [--assets-json PATH] [--max-download-bytes N]" >&2
  exit 2
}
stage="${1:-}"
[[ -n "$stage" ]] || usage
shift
run_id=""
allow_compute=false
allow_gpu=false
allow_network=false
allow_install=false
allow_download=false
asset_json=""
max_download_bytes=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --run-id) [[ "$#" -ge 2 ]] || usage; run_id="$2"; shift 2 ;;
    --allow-compute) allow_compute=true; shift ;;
    --allow-gpu) allow_gpu=true; shift ;;
    --allow-network) allow_network=true; shift ;;
    --allow-install) allow_install=true; shift ;;
    --allow-download) allow_download=true; shift ;;
    --assets-json) [[ "$#" -ge 2 ]] || usage; asset_json="$2"; shift 2 ;;
    --max-download-bytes) [[ "$#" -ge 2 ]] || usage; max_download_bytes="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$max_download_bytes" =~ ^[0-9]+$ ]] || usage
if "$allow_gpu" && ! "$allow_compute"; then
  echo "BLOCKED_AUTH: --allow-gpu also requires explicit --allow-compute" >&2
  exit 3
fi
if { "$allow_install" || "$allow_download"; } && ! "$allow_network"; then
  echo "BLOCKED_AUTH: installation/download requires explicit --allow-network" >&2
  exit 3
fi
if [[ "$stage" != "inspect" && "$stage" != "setup" && "$stage" != "asset_prepare" ]]; then
  [[ "$run_id" =~ ^[A-Za-z][A-Za-z0-9_-]{1,63}$ ]] || usage
fi

# Fail closed if configuration drifts from what the currently shipped runner
# actually implements. The local Agent reports a mismatch rather than
# silently ignoring the Workplan's own default configuration.
validate_defaults() {
  python -c 'import sys,yaml; p=yaml.safe_load(open("conf/experiment/wp23_precheck.yaml", encoding="utf-8"))["experiment"]; c=p["checks"]; i=p["inventory"]; ok=(i["sae_sites_per_model_max"]==2 and c["max_real_questions_per_pair"]==2 and c["max_seconds_per_worker"]==180 and c["max_tokens_per_worker"]==128 and c["default_task_smokes"]==["capture","intervene","generate","benchmark"] and c["allow_synthetic_pass"] is False); sys.exit(0 if ok else "BLOCKED_CONFIG: runner and declared WP23 defaults disagree")'
}
worker_args=()
if "$allow_compute"; then worker_args+=(--allow-compute); fi
if "$allow_gpu"; then worker_args+=(--allow-gpu); fi
invoke() {
  python -m scripts.workplan.wp023_orchestrator "$1" --run-id "$run_id" "${worker_args[@]}"
}
setup() {
  echo "WP23 Setup: checking installed dependencies; no download or GPU execution"
  python -m pip check
  python -c 'import sys,platform,json,importlib.util,shutil; print(json.dumps({"python":sys.version.split()[0],"platform":platform.platform(),"disk_free":shutil.disk_usage(".").free,"torch_installed":importlib.util.find_spec("torch") is not None},sort_keys=True))'
  validate_defaults
}
prepare_assets() {
  "$allow_network" || { echo "BLOCKED_AUTH: explicit --allow-network required to prepare missing assets" >&2; return 3; }
  ASSET_JSON="$asset_json" ALLOW_INSTALL="$allow_install" ALLOW_DOWNLOAD="$allow_download" MAX_DOWNLOAD_BYTES="$max_download_bytes" python -c 'import os,json,sys; from saebase.setup.assets import prepare_environment; path=os.environ["ASSET_JSON"]; assets=json.load(open(path,encoding="utf-8")) if path else []; result=prepare_environment(assets=assets,allow_network=True,allow_install=os.environ["ALLOW_INSTALL"]=="true",allow_download=os.environ["ALLOW_DOWNLOAD"]=="true",max_download_bytes=int(os.environ["MAX_DOWNLOAD_BYTES"])); print(json.dumps(result,ensure_ascii=False)); sys.exit(0 if result["status"]=="READY" else 3)'
}
resume_guard() {
  if "$allow_compute" && [[ -d "experiment/_studies/wp23_component_precheck/reports/$run_id/receipts" ]]; then
    if grep -lq '"code": "BLOCKED_AUTH"' "experiment/_studies/wp23_component_precheck/reports/$run_id/receipts/"*.json 2>/dev/null; then
      echo "BLOCKED_RESUME: previous authorization-blocked receipts are immutable. Start a new --run-id after authorization changes; never reuse their result." >&2
      return 3
    fi
  fi
}
run_pipeline() {
  validate_defaults
  local base="experiment/_studies/wp23_component_precheck/reports/$run_id"
  if [[ ! -f "$base/inventory.json" ]]; then
    invoke inventory
  fi
  resume_guard
  local step
  for step in dataset model sae compatibility task; do
    invoke "$step"
  done
  # Audit always runs even when some checks returned truthful BLOCKED rows.
  invoke audit
  verify_handoff
}
verify_handoff() {
  local base="experiment/_studies/wp23_component_precheck/reports/$run_id"
  local filename
  for filename in inventory.json targets.json component_checks.csv compatibility_checks.csv task_checks.csv coverage_summary.csv artifact_manifest.json audit.json archive_receipt.json workplan_completed.md; do
    if [[ ! -s "$base/$filename" ]]; then
      echo "BLOCKED_HANDOFF: missing or empty $base/$filename" >&2
      return 3
    fi
  done
  echo "WP23 execution report available at $base/workplan_completed.md"
  echo "Scientific acceptance and real native checks must be determined from receipts, not from this handoff check."
}
case "$stage" in
  setup) setup ;;
  asset_prepare) prepare_assets ;;
  inspect) validate_defaults; python -m scripts.workplan.wp023_orchestrator inspect ;;
  inventory|dataset|model|sae|compatibility|task|audit)
    validate_defaults
    resume_guard
    invoke "$stage"
    ;;
  handoff) validate_defaults; verify_handoff ;;
  run|resume) run_pipeline ;;
  *) usage ;;
esac
