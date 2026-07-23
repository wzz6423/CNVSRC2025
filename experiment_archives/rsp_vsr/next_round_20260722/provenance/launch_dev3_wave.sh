#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 WAVE_NUMBER" >&2
  exit 2
fi

wave="$1"
expected_commit="${DEV3_CODE_COMMIT:?DEV3_CODE_COMMIT must be the full commit hash}"
short_commit="${expected_commit:0:7}"
code_dir="/hy-tmp/rsp-vsr-dev3-${short_commit}/VSR"
output_root="/hy-tmp/experiments/rsp_vsr/chinese_lips_trainpool_dev3_active_feedback_${short_commit}"
manifest="/hy-tmp/datasets/manifests/chinese_lips_trainpool_dev3_revisit_120_176_183_seed42.jsonl"
python_bin="/hy-tmp/envs/rsp-vsr/bin/python"

[[ "$(git -C "${code_dir}/.." rev-parse HEAD)" == \
  "${expected_commit}" ]]
[[ "$(sha256sum "${manifest}" | cut -d' ' -f1)" == \
  "e3ecd63cd9b07f1df82bc198f6944a52fb92a660f1b3445308c14257dfc651de" ]]
[[ "$(sha256sum "${manifest}.meta.json" | cut -d' ' -f1)" == \
  "2b731e12d7f1b87f91d3fa047f10ec3cb1051427dd937838a779869c29c52e64" ]]
[[ "$(df --output=avail -BG /hy-tmp | tail -1 | tr -dc 0-9)" -gt 30 ]]

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

mkdir -p "${output_root}"
common=(
  "data_root_dir=/hy-tmp/cn_dataset/chinese_lips/chinese_lips"
  "stream_manifest=${manifest}"
  "checkpoint_path=/hy-tmp/rsp-vsr/VSR/pretrained_models/model_avg_cncvs_2_3_cnvsrc.pth"
  "plasticity.allow_growth=false"
  "plasticity.max_experts=1"
  "plasticity.feedback_update_strategy=full_sequence"
  "feedback.random_seed=42"
  "feedback.uncertainty_history_window=100"
  "feedback.noise_rate=0.0"
  "log_every=25"
  "checkpoint_every=100"
  "checkpoint_file_limit=3"
  "metrics_history_every=25"
  "checkpoint_selection_metric=mean_reliability"
  "seed=42"
)

launch() {
  local session="$1"
  local run_name="$2"
  local gpu="$3"
  shift 3
  local output_dir="${output_root}/${run_name}"
  if [[ -e "${output_dir}" ]]; then
    echo "Run directory already exists: ${output_dir}" >&2
    exit 6
  fi
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 7
  fi
  local command=(
    env
    "PYTHON_BIN=${python_bin}"
    "MAX_RESTARTS=3"
    bash
    scripts/run_continual_experiment.sh
    "${output_dir}"
    "${gpu}"
    "${common[@]}"
    "$@"
  )
  local command_text
  printf -v command_text '%q ' "${command[@]}"
  tmux new-session -d -s "${session}" -c "${code_dir}" "${command_text}"
}

case "${wave}" in
  1)
    launch \
      rsp_dev3_periodic \
      dev3_revisit_combined_periodic_feedback10_seed42 \
      0 \
      plasticity.mode=single_adapter \
      plasticity.non_feedback_updates_enabled=true \
      feedback.strategy=periodic \
      feedback.every=10
    launch \
      rsp_dev3_uncertainty \
      dev3_revisit_combined_uncertainty_feedback10_seed42 \
      1 \
      plasticity.mode=single_adapter \
      plasticity.non_feedback_updates_enabled=true \
      feedback.strategy=uncertainty \
      feedback.every=10
    ;;
  2)
    launch \
      rsp_dev3_random \
      dev3_revisit_combined_random_feedback10_seed42 \
      0 \
      plasticity.mode=single_adapter \
      plasticity.non_feedback_updates_enabled=true \
      feedback.strategy=random \
      feedback.every=10
    launch \
      rsp_dev3_static \
      dev3_revisit_static_seed42 \
      1 \
      plasticity.mode=static \
      plasticity.non_feedback_updates_enabled=false \
      feedback.strategy=periodic \
      feedback.every=0
    ;;
  3)
    launch \
      rsp_dev3_pseudo \
      dev3_revisit_pseudo_only_seed42 \
      0 \
      plasticity.mode=single_adapter \
      plasticity.non_feedback_updates_enabled=true \
      feedback.strategy=periodic \
      feedback.every=0
    launch \
      rsp_dev3_feedback_only \
      dev3_revisit_feedback_only_periodic_feedback10_seed42 \
      1 \
      plasticity.mode=single_adapter \
      plasticity.non_feedback_updates_enabled=false \
      feedback.strategy=periodic \
      feedback.every=10
    ;;
  *)
    echo "WAVE_NUMBER must be 1, 2, or 3" >&2
    exit 2
    ;;
esac

sleep 3
tmux ls
echo "${gpu_rows}"
