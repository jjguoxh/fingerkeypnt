#!/usr/bin/env bash
set -euo pipefail

# 切换到脚本所在目录
cd "$(dirname "$0")"

ENV_NAME="yolo-env"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"

  # 如果环境不存在则创建（可选）
  if ! conda env list | grep -q "^${ENV_NAME}\s"; then
    conda create -y -n "$ENV_NAME" -c conda-forge python=3.11
  fi

  # 安装/校验必需依赖（仅在缺失时安装）
  missing=0
  for pkg in pygame opencv-python mediapipe numpy; do
    if ! conda run -n "$ENV_NAME" python -c "import ${pkg%%-*}" >/dev/null 2>&1; then
      missing=1
    fi
  done
  if [ "$missing" -eq 1 ]; then
    echo "[信息] 检测到依赖缺失，正在安装: pygame opencv-python mediapipe numpy"
    conda run -n "$ENV_NAME" pip install pygame opencv-python mediapipe numpy || true
  fi

  # 运行游戏
  exec conda run -n "$ENV_NAME" python game2.py
else
  echo "未检测到conda，请确认Anaconda已正确安装并在PATH中。" >&2
  exit 1
fi