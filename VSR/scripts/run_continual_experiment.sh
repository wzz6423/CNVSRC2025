#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 OUTPUT_DIR GPU_ID [HYDRA_OVERRIDE ...]" >&2
  exit 2
fi

output_dir="$1"
gpu_id="$2"
shift 2

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
max_restarts="${MAX_RESTARTS:-3}"
restart_delay="${RESTART_DELAY_SECONDS:-30}"
health_interval="${HEALTH_INTERVAL_SECONDS:-60}"
stall_timeout="${STALL_TIMEOUT_SECONDS:-2700}"

mkdir -p "${output_dir}"
output_dir="$(cd "${output_dir}" && pwd)"
log_file="${output_dir}/run.log"
status_file="${output_dir}/supervisor_status.tsv"
checkpoint_file="${output_dir}/adaptation_state.pt"
result_file="${output_dir}/stream_results.jsonl"
summary_file="${output_dir}/summary.json"
worker_pid_file="${output_dir}/worker.pid"
metadata_file="${output_dir}/run_metadata.txt"
child_pid=""

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

file_size() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo 0
  elif stat -c %s "${path}" >/dev/null 2>&1; then
    stat -c %s "${path}"
  else
    stat -f %z "${path}"
  fi
}

summary_is_valid() {
  [[ -s "${summary_file}" ]] || return 1
  [[ -s "${checkpoint_file}" ]] || return 1
  [[ -s "${result_file}" ]] || return 1
  "${python_bin}" - "${summary_file}" "${checkpoint_file}" "${result_file}" <<'PY'
import json
import hashlib
import sys
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
required = {
    "samples",
    "mode",
    "stream_manifest",
    "base_checkpoint",
    "stream_state",
    "expert_bank",
}
if not isinstance(summary, dict) or not required.issubset(summary):
    raise SystemExit(1)
if not isinstance(summary["samples"], int) or summary["samples"] < 0:
    raise SystemExit(1)
checkpoint = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
if checkpoint.get("processed_samples") != summary["samples"]:
    raise SystemExit(1)
stream_state = summary["stream_state"]
if not isinstance(stream_state, dict) or checkpoint.get("stream_state") != stream_state:
    raise SystemExit(1)
manifest_path = Path(summary["stream_manifest"])
if sha256(manifest_path) != stream_state.get("manifest_sha256"):
    raise SystemExit(1)
if "manifest_metadata_sha256" in stream_state:
    metadata_path = Path(f"{manifest_path}.meta.json")
    if sha256(metadata_path) != stream_state["manifest_metadata_sha256"]:
        raise SystemExit(1)
if "target_vocab_sha256" in stream_state:
    if sha256("datamodule/char_units.txt") != stream_state["target_vocab_sha256"]:
        raise SystemExit(1)
if sha256(summary["base_checkpoint"]) != stream_state.get(
    "base_checkpoint_sha256"
):
    raise SystemExit(1)
with open(sys.argv[3], encoding="utf-8") as handle:
    result_rows = sum(1 for line in handle if line.strip())
if result_rows != summary["samples"]:
    raise SystemExit(1)
PY
}

write_status() {
  local state="$1"
  local attempt="$2"
  local detail="$3"
  local temporary="${status_file}.tmp"
  {
    printf 'state\t%s\n' "${state}"
    printf 'updated_at\t%s\n' "$(timestamp)"
    printf 'attempt\t%s\n' "${attempt}"
    printf 'gpu_id\t%s\n' "${gpu_id}"
    printf 'worker_pid\t%s\n' "${child_pid}"
    printf 'result_bytes\t%s\n' "$(file_size "${result_file}")"
    printf 'detail\t%s\n' "${detail}"
  } > "${temporary}"
  mv "${temporary}" "${status_file}"
}

cleanup() {
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill -TERM "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  rm -f "${worker_pid_file}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec 9>"${output_dir}/supervisor.lock"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
  echo "Another supervisor already owns ${output_dir}" >&2
  exit 3
fi

cd "${code_root}"

base_command=(
  "${python_bin}"
  -u
  continual_adapt.py
  "output_dir=${output_dir}"
  "$@"
)

printf -v command_text '%q ' "${base_command[@]}"
if [[ -f "${metadata_file}" ]]; then
  previous_command="$(sed -n 's/^command=//p' "${metadata_file}")"
  if [[ "${previous_command}" != "${command_text}" ]]; then
    write_status "failed" 0 "output directory belongs to another command"
    exit 4
  fi
else
  {
    printf 'started_at=%s\n' "$(timestamp)"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'gpu_id=%s\n' "${gpu_id}"
    printf 'python=%s\n' "$("${python_bin}" --version 2>&1)"
    printf 'command=%s\n' "${command_text}"
  } > "${metadata_file}"
fi

if summary_is_valid; then
  write_status "completed" 0 "summary already exists"
  exit 0
fi

attempt=0
while (( attempt <= max_restarts )); do
  attempt=$((attempt + 1))
  command=("${base_command[@]}")
  if [[ -s "${checkpoint_file}" && -s "${result_file}" ]]; then
    command+=("resume_adaptation_checkpoint=${checkpoint_file}")
  fi

  {
    printf '[%s] launch attempt %d/%d: ' \
      "$(timestamp)" "${attempt}" "$((max_restarts + 1))"
    printf '%q ' "${command[@]}"
    printf '\n'
  } >> "${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" RSP_VSR_DEVICE="cuda:0" \
    "${command[@]}" >> "${log_file}" 2>&1 &
  child_pid=$!
  printf '%s\n' "${child_pid}" > "${worker_pid_file}"
  write_status "running" "${attempt}" "worker launched"

  last_size="$(file_size "${result_file}")"
  last_progress="$(date +%s)"
  stalled=false
  while kill -0 "${child_pid}" 2>/dev/null; do
    sleep "${health_interval}"
    current_size="$(file_size "${result_file}")"
    now="$(date +%s)"
    if (( current_size > last_size )); then
      last_size="${current_size}"
      last_progress="${now}"
    fi
    if (( now - last_progress >= stall_timeout )); then
      stalled=true
      write_status "stalled" "${attempt}" "no result growth for ${stall_timeout}s"
      kill -TERM "${child_pid}" 2>/dev/null || true
      sleep 10
      kill -KILL "${child_pid}" 2>/dev/null || true
      break
    fi
    write_status "running" "${attempt}" "worker healthy"
  done

  set +e
  wait "${child_pid}"
  exit_code=$?
  set -e
  child_pid=""
  rm -f "${worker_pid_file}"

  if (( exit_code == 0 )) && summary_is_valid; then
    write_status "completed" "${attempt}" "summary written"
    exit 0
  fi

  if (( attempt > max_restarts )); then
    detail="worker exited with code ${exit_code}"
    if [[ "${stalled}" == true ]]; then
      detail="worker repeatedly stalled"
    fi
    write_status "failed" "${attempt}" "${detail}"
    if (( exit_code == 0 )); then
      exit_code=1
    fi
    exit "${exit_code}"
  fi

  write_status "restarting" "${attempt}" "worker exited with code ${exit_code}"
  sleep "${restart_delay}"
done
