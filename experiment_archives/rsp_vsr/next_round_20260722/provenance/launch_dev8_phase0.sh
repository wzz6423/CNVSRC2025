#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
  echo "Usage: $0" >&2
  exit 2
fi

expected_commit="d123cf3cca7b900be0c8baa6538fd6237081be14"
code_root="/hy-tmp/rsp-vsr-dev8-d123cf3-bundle"
code_dir="${code_root}/VSR"
python_bin="/hy-tmp/envs/rsp-vsr-lm-qwen25/bin/python"
model_dir="/hy-tmp/models/Qwen2.5-0.5B-Instruct-7ae5576"
output_root="/hy-tmp/experiments/rsp_vsr/dev8_qwen_nbest_phase0_d123cf3"
dev6_input="/hy-tmp/experiments/rsp_vsr/chinese_lips_trainpool_dev6_grounded_repair_8f28af0/dev6_revisit_replay_adapter_nbest10_periodic_feedback10_seed42/stream_results.jsonl"
dev7_input="/hy-tmp/experiments/rsp_vsr/chinese_lips_trainpool_dev7_counterfactual_margin_b9d6d49/dev7_revisit_replay_adapter_nbest10_periodic_feedback10_seed42/stream_results.jsonl"

[[ "$(sha256sum "${code_dir}/scripts/evaluate_llm_nbest_selector.py" | cut -d' ' -f1)" == \
  "4d13c4bfb9eabe5929911425d6321c71fe5906a94a6267d3c0dc4bc9294f5517" ]]
[[ "$(sha256sum "${code_dir}/scripts/smoke_llm_nbest_selector.py" | cut -d' ' -f1)" == \
  "154346f2c1fc559736071be3e93d62bbe4b7fc8bbad72e99228a863782f837e5" ]]
[[ -x "${python_bin}" ]]
[[ "$(${python_bin} -c 'import transformers; print(transformers.__version__)')" == "4.46.3" ]]
[[ "$(sha256sum "${model_dir}/config.json" | cut -d' ' -f1)" == \
  "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45" ]]
[[ -s "${model_dir}/model.safetensors" ]]
[[ "$(sha256sum "${dev6_input}" | cut -d' ' -f1)" == \
  "2570f297331281b6dec5389546bad75de36d35de51c86bdc65d32903c1035922" ]]
[[ "$(sha256sum "${dev7_input}" | cut -d' ' -f1)" == \
  "a7eff26328a8ba136e575efa5c775a17e941743297a74014adb5f4303c6f3c69" ]]
[[ "$(wc -l < "${dev6_input}" | tr -d ' ')" == "356" ]]
[[ "$(wc -l < "${dev7_input}" | tr -d ' ')" == "625" ]]
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

for session in rsp_dev8_qwen_dev6 rsp_dev8_qwen_dev7; do
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 6
  fi
done

mkdir -p "${output_root}/dev6" "${output_root}/dev7"

launch() {
  local session="$1"
  local gpu="$2"
  local name="$3"
  local input="$4"
  local samples="$5"
  local output_dir="${output_root}/${name}"
  local command=(
    env
    "CUDA_VISIBLE_DEVICES=${gpu}"
    "PYTHONUNBUFFERED=1"
    "TOKENIZERS_PARALLELISM=false"
    "${python_bin}"
    scripts/evaluate_llm_nbest_selector.py
    --input "${input}"
    --output-dir "${output_dir}"
    --model-dir "${model_dir}"
    --model-id "Qwen/Qwen2.5-0.5B-Instruct"
    --model-revision "7ae557604adf67be50417f59c2c2f167def9a775"
    --code-commit "${expected_commit}"
    --device cuda:0
    --expected-samples "${samples}"
    --top-k 10
    --batch-size 8
    --max-new-tokens 8
    --bootstrap-samples 10000
    --seed 42
  )
  local command_text
  printf -v command_text '%q ' "${command[@]}"
  tmux new-session -d -s "${session}" -c "${code_dir}" \
    "${command_text} > '${output_dir}/run.log' 2>&1"
}

launch rsp_dev8_qwen_dev6 0 dev6 "${dev6_input}" 356
launch rsp_dev8_qwen_dev7 1 dev7 "${dev7_input}" 625

sleep 5
tmux has-session -t rsp_dev8_qwen_dev6
tmux has-session -t rsp_dev8_qwen_dev7
tmux ls
nvidia-smi \
  --query-gpu=index,temperature.gpu,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
