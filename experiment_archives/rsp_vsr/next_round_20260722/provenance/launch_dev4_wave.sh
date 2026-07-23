#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 WAVE_NUMBER" >&2
  exit 2
fi

wave="$1"
expected_commit="4191437d734f34cb524b049ba42415ec224a7ecb"
code_dir="/hy-tmp/rsp-vsr-baselines-4191437/VSR"
output_root="/hy-tmp/experiments/rsp_vsr/chinese_lips_trainpool_dev4_strong_baselines_4191437"
manifest="/hy-tmp/datasets/manifests/chinese_lips_trainpool_dev4_revisit_128_047_202_seed42.jsonl"
source_csv="/hy-tmp/datasets/manifests/chinese_lips_trainpool_dev4_128_047_202_seed42_source.csv"
base_checkpoint="/hy-tmp/rsp-vsr/VSR/pretrained_models/model_avg_cncvs_2_3_cnvsrc.pth"
python_bin="/hy-tmp/envs/rsp-vsr/bin/python"

[[ "$(git -C "${code_dir}/.." rev-parse HEAD)" == "${expected_commit}" ]]
[[ "$(sha256sum "${source_csv}" | cut -d' ' -f1)" == \
  "c6ee6d9c6ebf7809a98c4a9a145ed0d0da68a68841624aedc626ac058bc0ed8a" ]]
[[ "$(sha256sum "${manifest}" | cut -d' ' -f1)" == \
  "a3b1a5842d05ec2caf09e63eb087a6d0623770ce4edcaeca9e9d9b14eb1bf132" ]]
[[ "$(sha256sum "${manifest}.meta.json" | cut -d' ' -f1)" == \
  "a347d97f547498788847c0c9ed9247b7d67cf324d6e5f7cb760ca370542bc2f2" ]]
[[ "$(sha256sum "${base_checkpoint}" | cut -d' ' -f1)" == \
  "577cd9558eea111683a406bc25d69c7161cdb79534c2273fc0d0f044c356231c" ]]
[[ "$(sha256sum "${code_dir}/datamodule/char_units.txt" | cut -d' ' -f1)" == \
  "635e12ebb5f7dcd60637a4f3c329cd543f1e0e34aa4a6d62ba87185c3666aae0" ]]
[[ "$(wc -l < "${manifest}" | tr -d ' ')" == "700" ]]
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
  "checkpoint_path=${base_checkpoint}"
  "require_manifest_metadata=true"
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
  "checkpoint_selection_mode=max"
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
      rsp_dev4_static \
      dev4_revisit_static_seed42 \
      0 \
      plasticity.mode=static \
      plasticity.adaptation_objective=rsp \
      plasticity.non_feedback_updates_enabled=false \
      feedback.strategy=periodic \
      feedback.every=0
    launch \
      rsp_dev4_bn_tent \
      dev4_revisit_bn_tent_nofeedback_seed42 \
      1 \
      plasticity.mode=parameter_adaptation \
      plasticity.adaptation_objective=bn_tent \
      plasticity.rollback_enabled=false \
      plasticity.non_feedback_updates_enabled=true \
      feedback.strategy=periodic \
      feedback.every=0
    ;;
  2)
    launch \
      rsp_dev4_eta \
      dev4_revisit_eta_nofeedback_seed42 \
      0 \
      plasticity.mode=parameter_adaptation \
      plasticity.adaptation_objective=eta \
      plasticity.rollback_enabled=false \
      plasticity.non_feedback_updates_enabled=true \
      feedback.strategy=periodic \
      feedback.every=0
    launch \
      rsp_dev4_incumbent \
      dev4_revisit_combined_periodic_feedback10_seed42 \
      1 \
      plasticity.mode=single_adapter \
      plasticity.adaptation_objective=rsp \
      plasticity.non_feedback_updates_enabled=true \
      feedback.strategy=periodic \
      feedback.every=10
    ;;
  3)
    launch \
      rsp_dev4_online_lora \
      dev4_revisit_online_lora_periodic_feedback10_seed42 \
      0 \
      plasticity.mode=parameter_adaptation \
      plasticity.adaptation_objective=online_lora \
      plasticity.rollback_enabled=false \
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
