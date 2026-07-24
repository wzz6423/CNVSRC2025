#!/usr/bin/env bash
set -euo pipefail

expected_commit="8f28af0253c7c4b5a04736e559ba368217a7cc76"
code_dir="/hy-tmp/rsp-vsr-dev6-8f28af0/VSR"
output_root="/hy-tmp/experiments/rsp_vsr/chinese_lips_trainpool_dev6_grounded_repair_8f28af0"
run_name="dev6_revisit_replay_adapter_nbest10_periodic_feedback10_seed42"
session="rsp_dev6_nbest10"
manifest="/hy-tmp/datasets/manifests/chinese_lips_trainpool_dev5_revisit_071_126_045_seed42.jsonl"
source_csv="/hy-tmp/datasets/manifests/chinese_lips_trainpool_dev5_071_126_045_seed42_source.csv"
manifest_audit="/hy-tmp/datasets/manifests/chinese_lips_trainpool_dev5_revisit_071_126_045_seed42.audit.json"
base_checkpoint="/hy-tmp/rsp-vsr/VSR/pretrained_models/model_avg_cncvs_2_3_cnvsrc.pth"
python_bin="/hy-tmp/envs/rsp-vsr/bin/python"
gpu=0

[[ "$(git -C "${code_dir}/.." rev-parse HEAD)" == "${expected_commit}" ]]
[[ -z "$(git -C "${code_dir}/.." status --porcelain)" ]]
[[ "$(sha256sum "${source_csv}" | cut -d' ' -f1)" == \
  "fe44c1bc4a99b636d3fcc07b3ab999eb9484c62a4da52cebf6efac88b453eef5" ]]
[[ "$(sha256sum "${manifest}" | cut -d' ' -f1)" == \
  "8c8e967e7076562da70d47f45883ead84c5c2dd2ef47f69412467db2e89ebf56" ]]
[[ "$(sha256sum "${manifest}.meta.json" | cut -d' ' -f1)" == \
  "27d96cedeba419350abb2daabd1ea9e2be03127d62b0eaf1badd58be7d9421c3" ]]
[[ "$(sha256sum "${manifest_audit}" | cut -d' ' -f1)" == \
  "6ea95d0b4c6c7a69a4d8d1beac2c593a85716a28dfb6f06b061ae1daf095c750" ]]
[[ "$(sha256sum "${base_checkpoint}" | cut -d' ' -f1)" == \
  "577cd9558eea111683a406bc25d69c7161cdb79534c2273fc0d0f044c356231c" ]]
[[ "$(sha256sum "${code_dir}/datamodule/char_units.txt" | cut -d' ' -f1)" == \
  "635e12ebb5f7dcd60637a4f3c329cd543f1e0e34aa4a6d62ba87185c3666aae0" ]]
[[ "$(wc -l < "${source_csv}" | tr -d ' ')" == "681" ]]
[[ "$(wc -l < "${manifest}" | tr -d ' ')" == "681" ]]
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

output_dir="${output_root}/${run_name}"
if [[ -e "${output_dir}" ]]; then
  echo "Run directory already exists: ${output_dir}" >&2
  exit 6
fi
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists: ${session}" >&2
  exit 7
fi

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
  "plasticity.mode=single_adapter"
  "plasticity.adapter_type=bottleneck"
  "plasticity.adaptation_objective=rsp"
  "plasticity.non_feedback_updates_enabled=true"
  "feedback.strategy=periodic"
  "feedback.every=10"
  "decoder.nbest_size=10"
)

mkdir -p "${output_root}"
command=(
  env
  "PYTHON_BIN=${python_bin}"
  "MAX_RESTARTS=3"
  bash
  scripts/run_continual_experiment.sh
  "${output_dir}"
  "${gpu}"
  "${common[@]}"
)
printf -v command_text '%q ' "${command[@]}"
tmux new-session -d -s "${session}" -c "${code_dir}" "${command_text}"

sleep 3
tmux ls
echo "${gpu_rows}"
