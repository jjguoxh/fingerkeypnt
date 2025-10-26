#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MACOS_DIR="$ROOT_DIR/macos"
BUILD_DIR="$MACOS_DIR/build"
mkdir -p "$BUILD_DIR"

# 检查开发工具
if ! xcode-select -p >/dev/null 2>&1; then
  echo "[错误] 需要安装 Xcode 命令行工具：xcode-select --install" >&2
  exit 1
fi

# OpenCV 检测（可选）
OPENCV_CFLAGS=""
OPENCV_LIBS=""
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists opencv4; then
  OPENCV_CFLAGS="$(pkg-config --cflags opencv4)"
  OPENCV_LIBS="$(pkg-config --libs opencv4)"
  echo "[信息] 使用 pkg-config 提供的 OpenCV 配置"
else
  # Homebrew 典型路径回退
  if [ -d "/opt/homebrew/include/opencv4" ] || [ -d "/usr/local/include/opencv4" ]; then
    INC_DIR="/opt/homebrew/include"
    LIB_DIR="/opt/homebrew/lib"
    if [ ! -d "$INC_DIR" ]; then INC_DIR="/usr/local/include"; fi
    if [ ! -d "$LIB_DIR" ]; then LIB_DIR="/usr/local/lib"; fi
    OPENCV_CFLAGS="-I$INC_DIR"
    OPENCV_LIBS="-L$LIB_DIR -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_highgui -lopencv_dnn"
    echo "[信息] 使用 Homebrew 回退 OpenCV 路径: $INC_DIR / $LIB_DIR"
  else
    echo "[警告] 未检测到 OpenCV，检测功能将不可用（应用仍可运行）。"
  fi
fi

# 编译 ObjC++ 封装
clang++ -std=c++17 -ObjC++ -fobjc-arc \
  -I"$MACOS_DIR" $OPENCV_CFLAGS \
  -framework Foundation -framework AVFoundation \
  -c "$MACOS_DIR/DetectorWrapper.mm" -o "$BUILD_DIR/DetectorWrapper.o"

# 编译并链接 Swift 代码
swiftc \
  "$MACOS_DIR/AppDelegate.swift" \
  "$MACOS_DIR/ViewController.swift" \
  -import-objc-header "$MACOS_DIR/GestureIdentify-Bridging-Header.h" \
  "$BUILD_DIR/DetectorWrapper.o" \
  -framework Cocoa -framework AVFoundation \
  $OPENCV_LIBS \
  -Xlinker -lc++ \
  -o "$BUILD_DIR/HandDetectionApp"

# 运行，传入项目根路径用于加载 models/
PROJECT_ROOT="$ROOT_DIR" "$BUILD_DIR/HandDetectionApp"