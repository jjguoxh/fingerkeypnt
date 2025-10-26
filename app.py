import os
import re
import time
from pathlib import Path
from typing import Tuple, List, Optional

import streamlit as st
import cv2
import numpy as np
import requests

# Try to import MediaPipe; fall back gracefully if unavailable
HAS_MEDIAPIPE = True
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
except Exception:
    HAS_MEDIAPIPE = False
    mp_hands = None
    mp_drawing = None
    mp_styles = None

PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
DOWNLOAD_SCRIPT_URL = "https://raw.githubusercontent.com/cansik/yolo-hand-detection/master/models/download-models.sh"


def ensure_models() -> Tuple[Optional[Path], Optional[Path], List[str]]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def find_best_pair() -> Tuple[Optional[Path], Optional[Path]]:
        cfgs = list(MODELS_DIR.glob("*.cfg"))
        weights = list(MODELS_DIR.glob("*.weights"))
        if not cfgs or not weights:
            return None, None
        order = [
            "cross-hands-yolov4-tiny",
            "cross-hands-tiny-prn",
            "cross-hands-tiny",
            "cross-hands",
        ]
        for name in order:
            c = MODELS_DIR / f"{name}.cfg"
            w = MODELS_DIR / f"{name}.weights"
            if c.exists() and w.exists():
                return c, w
        return sorted(cfgs)[0], sorted(weights)[0]

    cfg_path, weights_path = find_best_pair()
    if cfg_path and weights_path and cfg_path.exists() and weights_path.exists():
        names_path = next(MODELS_DIR.glob("*.names"), None)
        class_names = ["hand"]
        if names_path and names_path.exists():
            try:
                class_names = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]
            except Exception:
                pass
        return cfg_path, weights_path, class_names

    try:
        r = requests.get(DOWNLOAD_SCRIPT_URL, timeout=30)
        r.raise_for_status()
        script = r.text
        urls = list(set(re.findall(r"https?://[^\s'\"]+", script)))
        file_urls = [u for u in urls if any(u.endswith(ext) for ext in [".cfg", ".weights", ".names"])]
        for url in file_urls:
            fname = url.split("/")[-1]
            dst = MODELS_DIR / fname
            if dst.exists():
                continue
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        cfg_path, weights_path = find_best_pair()
    except Exception as e:
        st.error(f"下载模型文件失败: {e}")
        return None, None, ["hand"]

    names_path = next(MODELS_DIR.glob("*.names"), None)
    class_names = ["hand"]
    if names_path and names_path.exists():
        try:
            class_names = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]
        except Exception:
            pass

    return cfg_path, weights_path, class_names


def load_net(cfg_path: Path, weights_path: Path) -> Tuple[cv2.dnn_Net, List[str]]:
    net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    out_names = net.getUnconnectedOutLayersNames()
    return net, out_names


def letterbox(image: np.ndarray, new_shape: Tuple[int, int] = (416, 416), color=(114, 114, 114)):
    h, w = image.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = new_shape[1] - nw
    pad_h = new_shape[0] - nh
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)


def detect(frame: np.ndarray, net: cv2.dnn_Net, out_names: List[str], conf_thres: float = 0.35, nms_thres: float = 0.45, input_size: int = 416) -> Tuple[List[List[int]], List[float], List[int]]:
    H, W = frame.shape[:2]
    lb_image, r, (pad_x, pad_y) = letterbox(frame, (input_size, input_size))
    blob = cv2.dnn.blobFromImage(lb_image, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(out_names)

    boxes = []
    confidences = []
    class_ids = []

    LBH, LBW = lb_image.shape[:2]

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            class_score = float(scores[class_id])
            objectness = float(detection[4])
            confidence = objectness * class_score
            if confidence > conf_thres:
                box = detection[0:4] * np.array([LBW, LBH, LBW, LBH])
                centerX, centerY, width, height = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                x0 = int((x - pad_x) / r)
                y0 = int((y - pad_y) / r)
                x1 = int((x + width - pad_x) / r)
                y1 = int((y + height - pad_y) / r)
                x0 = max(0, min(W - 1, x0))
                y0 = max(0, min(H - 1, y0))
                x1 = max(0, min(W - 1, x1))
                y1 = max(0, min(H - 1, y1))
                boxes.append([x0, y0, x1 - x0, y1 - y0])
                confidences.append(confidence)
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, nms_thres)
    if len(idxs) > 0:
        idxs = idxs.flatten().tolist()
        boxes = [boxes[i] for i in idxs]
        confidences = [confidences[i] for i in idxs]
        class_ids = [class_ids[i] for i in idxs]
    else:
        boxes, confidences, class_ids = [], [], []

    return boxes, confidences, class_ids


def draw_detections(frame: np.ndarray, boxes: List[List[int]], confidences: List[float], class_ids: List[int], class_names: List[str]) -> np.ndarray:
    for (x, y, w, h), conf, cid in zip(boxes, confidences, class_ids):
        label = class_names[cid] if cid < len(class_names) else "hand"
        color = (0, 200, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{label}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(frame, text, (x + 2, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0


def bbox_from_landmarks(landmarks, W, H):
    xs = [int(l.x * W) for l in landmarks.landmark]
    ys = [int(l.y * H) for l in landmarks.landmark]
    x0, y0 = max(0, min(xs)), max(0, min(ys))
    x1, y1 = min(W - 1, max(xs)), min(H - 1, max(ys))
    return [x0, y0, x1 - x0, y1 - y0]


st.set_page_config(page_title="手部检测 (YOLO + Hands)", layout="wide")

st.title("使用 YOLO 检测 + 手指关键点绘制")
st.caption("若本地无手部YOLO模型，将自动从网络下载。")

placeholder = st.empty()
status_text = st.sidebar.empty()

conf_thres = st.sidebar.slider("检测置信度阈值", 0.1, 0.9, 0.35, 0.05)
nms_thres = st.sidebar.slider("NMS阈值", 0.1, 0.9, 0.45, 0.05)
input_size = st.sidebar.slider("YOLO输入尺寸", 320, 640, 416, 32)

# Landmarks controls
if HAS_MEDIAPIPE:
    enable_landmarks = st.sidebar.checkbox("绘制手指关键点(21)", value=True)
    max_hands = st.sidebar.slider("最大手数", 1, 2, 1)
    match_iou_thres = st.sidebar.slider("关键点匹配IOU阈值", 0.0, 0.8, 0.2, 0.05)
else:
    enable_landmarks = False
    st.sidebar.warning("当前环境不支持安装 MediaPipe (Python 3.13)。可切换到 Python 3.10/3.11 以启用关键点。")

# Model selection
cfg_path, weights_path, class_names = ensure_models()
if not cfg_path or not weights_path:
    st.stop()

available_models = []
for stem in ["cross-hands-yolov4-tiny", "cross-hands-tiny-prn", "cross-hands-tiny", "cross-hands"]:
    c = MODELS_DIR / f"{stem}.cfg"
    w = MODELS_DIR / f"{stem}.weights"
    if c.exists() and w.exists():
        available_models.append((stem, c, w))

model_names = [m[0] for m in available_models]
selected_name = st.sidebar.selectbox("选择YOLO模型", model_names, index=0)
for name, c, w in available_models:
    if name == selected_name:
        cfg_path, weights_path = c, w
        break

net, out_names = load_net(cfg_path, weights_path)
status_text.info(f"使用模型: {cfg_path.name} / {weights_path.name}")

start = st.sidebar.button("开始检测")
stop = st.sidebar.button("停止检测")

if "running" not in st.session_state:
    st.session_state["running"] = False

if start:
    st.session_state["running"] = True
if stop:
    st.session_state["running"] = False

cap = None
try:
    if st.session_state["running"]:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("无法打开摄像头。请检查权限或设备。")
        else:
            fps_time = time.time()
            frame_win = placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="BGR")
            # Init hands once per run for performance
            hands = None
            if HAS_MEDIAPIPE and enable_landmarks:
                hands = mp_hands.Hands(static_image_mode=False, max_num_hands=max_hands, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            while st.session_state["running"]:
                ret, frame = cap.read()
                if not ret:
                    st.warning("读取摄像头帧失败。")
                    break
                H, W = frame.shape[:2]
                boxes, confs, cids = detect(frame, net, out_names, conf_thres=conf_thres, nms_thres=nms_thres, input_size=input_size)
                vis = draw_detections(frame.copy(), boxes, confs, cids, class_names)

                if HAS_MEDIAPIPE and enable_landmarks and hands is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = hands.process(rgb)
                    if res.multi_hand_landmarks:
                        for hand_landmarks in res.multi_hand_landmarks:
                            hb = bbox_from_landmarks(hand_landmarks, W, H)
                            draw_this = True
                            if boxes:
                                overlaps = [iou(hb, b) for b in boxes]
                                if max(overlaps) < match_iou_thres:
                                    draw_this = False
                            if draw_this:
                                mp_drawing.draw_landmarks(
                                    vis,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_styles.get_default_hand_landmarks_style(),
                                    mp_styles.get_default_hand_connections_style(),
                                )

                now = time.time()
                fps = 1.0 / (now - fps_time) if (now - fps_time) > 0 else 0.0
                fps_time = now
                cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)
                frame_win.image(vis, channels="BGR")
                time.sleep(0.001)
            if hands is not None:
                hands.close()
    else:
        st.info("点击左侧 ‘开始检测’ 按钮以启动摄像头检测。")
finally:
    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()