#!/usr/bin/env bash
set -euo pipefail

HF_REPO="ReflectionL/CNVSRC2025Baseline"
MODELSCOPE_REPO="PaintedVeil/CNVSRC2025Baseline"
MODEL_FILE="${1:-model_avg_cncvs_4s_30s.pth}"
OUTPUT_DIR="${2:-pretrained_models}"

mkdir -p "${OUTPUT_DIR}"
downloaded=false
if command -v hf >/dev/null 2>&1; then
  if hf download "${HF_REPO}" "${MODEL_FILE}" --local-dir "${OUTPUT_DIR}"; then
    downloaded=true
  else
    echo "Hugging Face 下载失败，尝试 ModelScope 镜像。" >&2
  fi
elif command -v huggingface-cli >/dev/null 2>&1; then
  if huggingface-cli download "${HF_REPO}" "${MODEL_FILE}" --local-dir "${OUTPUT_DIR}"; then
    downloaded=true
  else
    echo "Hugging Face 下载失败，尝试 ModelScope 镜像。" >&2
  fi
fi

if [[ "${downloaded}" == false ]] && command -v modelscope >/dev/null 2>&1; then
  modelscope download "${MODELSCOPE_REPO}" "${MODEL_FILE}" --local-dir "${OUTPUT_DIR}"
  downloaded=true
elif [[ "${downloaded}" == false ]] && command -v uvx >/dev/null 2>&1; then
  uvx --from modelscope modelscope download \
    "${MODELSCOPE_REPO}" "${MODEL_FILE}" --local-dir "${OUTPUT_DIR}"
  downloaded=true
fi

if [[ "${downloaded}" == false ]]; then
  echo "未找到可用的 hf、huggingface-cli、modelscope 或 uvx 下载工具。" >&2
  exit 1
fi

expected_sha256=""
case "${MODEL_FILE}" in
  model_avg_cncvs_4s_30s.pth)
    expected_sha256="79cb59044e925ce7583b46474ee17c84f938ea088861d33e4340614e1436104f"
    ;;
  model_avg_cncvs_2_3_cnvsrc.pth)
    expected_sha256="577cd9558eea111683a406bc25d69c7161cdb79534c2273fc0d0f044c356231c"
    ;;
esac

if [[ -n "${expected_sha256}" ]]; then
  output_path="${OUTPUT_DIR}/${MODEL_FILE}"
  if command -v shasum >/dev/null 2>&1; then
    actual_sha256="$(shasum -a 256 "${output_path}" | awk '{print $1}')"
  elif command -v sha256sum >/dev/null 2>&1; then
    actual_sha256="$(sha256sum "${output_path}" | awk '{print $1}')"
  else
    echo "未找到 shasum 或 sha256sum，跳过权重校验。" >&2
    exit 0
  fi
  if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    echo "权重 SHA-256 校验失败：${MODEL_FILE}" >&2
    exit 1
  fi
  echo "权重 SHA-256 校验通过：${MODEL_FILE}"
fi
