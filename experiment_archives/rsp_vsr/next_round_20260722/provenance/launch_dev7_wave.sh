#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
  echo "Usage: $0" >&2
  exit 2
fi

expected_commit="b9d6d499dfe88703cf45a661bf4463880cf0a632"
code_dir="/hy-tmp/rsp-vsr-dev7-b9d6d49/VSR"
output_root="/hy-tmp/experiments/rsp_vsr/chinese_lips_trainpool_dev7_counterfactual_margin_b9d6d49"
manifest="/hy-tmp/datasets/manifests/chinese_lips_trainpool_dev7_revisit_188_011_036_seed42.jsonl"
source_csv="/hy-tmp/datasets/manifests/chinese_lips_trainpool_dev7_188_011_036_seed42_source.csv"
manifest_audit="/hy-tmp/datasets/manifests/chinese_lips_trainpool_dev7_revisit_188_011_036_seed42.audit.json"
base_checkpoint="/hy-tmp/rsp-vsr/VSR/pretrained_models/model_avg_cncvs_2_3_cnvsrc.pth"
python_bin="/hy-tmp/envs/rsp-vsr/bin/python"

[[ "$(git -C "${code_dir}/.." rev-parse HEAD)" == "${expected_commit}" ]]
[[ -z "$(git -C "${code_dir}/.." status --porcelain)" ]]
[[ "$(sha256sum "${source_csv}" | cut -d' ' -f1)" == \
  "7e4ecd49a13c4aaca4de8a35ffe0d267497684b9e284c6bd9c479c18303aa243" ]]
[[ "$(sha256sum "${manifest}" | cut -d' ' -f1)" == \
  "22e94cffece7f496219225058c4547ec038f4046eb2f794a2cf6187d299467b8" ]]
[[ "$(sha256sum "${manifest}.meta.json" | cut -d' ' -f1)" == \
  "108649b9c751bda793f34ea4d9b7840c483ee996b9519fb8e2b69cf93800bedf" ]]
[[ "$(sha256sum "${manifest_audit}" | cut -d' ' -f1)" == \
  "54f29e6b0d6ead631cf3d7e2b013e3532c13adeec14f9019e221b142470be6f7" ]]
[[ "$(sha256sum "${base_checkpoint}" | cut -d' ' -f1)" == \
  "577cd9558eea111683a406bc25d69c7161cdb79534c2273fc0d0f044c356231c" ]]
[[ "$(sha256sum "${code_dir}/datamodule/char_units.txt" | cut -d' ' -f1)" == \
  "635e12ebb5f7dcd60637a4f3c329cd543f1e0e34aa4a6d62ba87185c3666aae0" ]]
[[ "$(wc -l < "${source_csv}" | tr -d ' ')" == "625" ]]
[[ "$(wc -l < "${manifest}" | tr -d ' ')" == "625" ]]
[[ "$(df --output=avail -BG /hy-tmp | tail -1 | tr -dc 0-9)" -gt 30 ]]
[[ ! -e "${output_root}" ]]

gpu_rows="$(
  nvidia-smi \
    --query-gpu=index,temperature.gpu,memory.used,utilization.gpu \
    --format=csv,noheader,nounits
)"
if ! awk -F, '
  {
    gsub(/ /, "", $2)
    gsub(/ /, "", $3)
    gsub(/ /, "", $4)
    if (($2 + 0) >= 78 || ($3 + 0) >= 1024 || ($4 + 0) >= 80) bad = 1
  }
  END { exit bad }
' <<<"${gpu_rows}"; then
  echo "GPU preflight failed:" >&2
  echo "${gpu_rows}" >&2
  exit 4
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -Eq '[0-9]'; then
  echo "GPU compute process already running" >&2
  exit 5
fi

for session in rsp_dev7_replay rsp_dev7_counterfactual; do
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 6
  fi
done

mkdir -p "${output_root}"
common=(
  "data_root_dir=/hy-tmp/cn_dataset/chinese_lips/chinese_lips"
  "stream_manifest=${manifest}"
  "checkpoint_path=${base_checkpoint}"
  "require_manifest_metadata=true"
  "decoder.beam_size=12"
  "decoder.ctc_weight=0.3"
  "decoder.nbest_size=10"
  "plasticity.mode=single_adapter"
  "plasticity.adapter_type=bottleneck"
  "plasticity.adaptation_objective=rsp"
  "plasticity.allow_growth=false"
  "plasticity.max_experts=1"
  "plasticity.feedback_update_strategy=full_sequence"
  "plasticity.non_feedback_updates_enabled=true"
  "plasticity.rollback_enabled=true"
  "plasticity.counterfactual_margin.margin=0.2"
  "plasticity.counterfactual_margin.weight=0.25"
  "plasticity.counterfactual_margin.rollback_tolerance=0.000001"
  "feedback.strategy=periodic"
  "feedback.every=10"
  "feedback.random_seed=42"
  "feedback.uncertainty_history_window=100"
  "feedback.noise_rate=0.0"
  "log_every=25"
  "checkpoint_every=100"
  "checkpoint_file_limit=3"
  "metrics_history_every=25"
  "checkpoint_selection_metric=mean_reliability"
  "checkpoint_selection_mode=max"
  "seed=42"
)

launch() {
  local session="$1"
  local run_name="$2"
  local gpu="$3"
  local counterfactual_enabled="$4"
  local output_dir="${output_root}/${run_name}"
  local command=(
    env
    "PYTHON_BIN=${python_bin}"
    "MAX_RESTARTS=0"
    "HEALTH_INTERVAL_SECONDS=30"
    bash
    scripts/run_continual_experiment.sh
    "${output_dir}"
    "${gpu}"
    "${common[@]}"
    "plasticity.counterfactual_margin.enabled=${counterfactual_enabled}"
  )
  local command_text
  printf -v command_text '%q ' "${command[@]}"
  tmux new-session -d -s "${session}" -c "${code_dir}" "${command_text}"
}

launch \
  rsp_dev7_replay \
  dev7_revisit_replay_adapter_nbest10_periodic_feedback10_seed42 \
  0 \
  false
launch \
  rsp_dev7_counterfactual \
  dev7_revisit_counterfactual_margin_nbest10_periodic_feedback10_seed42 \
  1 \
  true

sleep 3
tmux has-session -t rsp_dev7_replay
tmux has-session -t rsp_dev7_counterfactual
tmux ls
nvidia-smi \
  --query-gpu=index,temperature.gpu,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
