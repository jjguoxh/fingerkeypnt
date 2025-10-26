#!/usr/bin/env bash
set -euo pipefail

# 切换到脚本所在目录
cd "$(dirname "$0")"

ENV_NAME="yolo-env"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  if ! conda env list | grep -q "^${ENV_NAME}\s"; then
    conda create -y -n "$ENV_NAME" -c conda-forge python=3.11
    conda install -y -n "$ENV_NAME" -c conda-forge opencv numpy requests pyside6 streamlit
  fi
  # 安装mediapipe（若未安装）
  if ! conda run -n "$ENV_NAME" python -c "import mediapipe" >/dev/null 2>&1; then
    conda run -n "$ENV_NAME" pip install mediapipe || true
  fi
  # 启动Qt应用（使用PySide6后端）
  export QT_MAC_WANTS_LAYER=1
  exec conda run -n "$ENV_NAME" python qt_app.py
else
  echo "未检测到conda，请确认Anaconda已正确安装并在PATH中。" >&2
  exit 1
fi