import os
import re
import sys
import time
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
import requests

# Try PyQt5 first, fallback to PySide6
QtBackend = "PyQt5"
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton,
        QComboBox, QDoubleSpinBox, QSpinBox, QVBoxLayout, QHBoxLayout,
        QGroupBox, QCheckBox, QTextEdit
    )
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtGui import QImage, QPixmap
except Exception:
    QtBackend = "PySide6"
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton,
        QComboBox, QDoubleSpinBox, QSpinBox, QVBoxLayout, QHBoxLayout,
        QGroupBox, QCheckBox, QTextEdit
    )
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QImage, QPixmap

PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
DOWNLOAD_SCRIPT_URL = "https://raw.githubusercontent.com/cansik/yolo-hand-detection/master/models/download-models.sh"


# --- Model utils ---
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

    # Try to download models from download script
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
        print(f"下载模型文件失败: {e}")
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

    boxes, confidences, class_ids = [], [], []
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


# --- Qt GUI ---
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手部检测 - PyQt + YOLO")
        self.resize(980, 640)

        self.video_label = QLabel("摄像头画面")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#222;color:#aaa;")

        # Controls
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(240)
        self.input_spin = QSpinBox()
        self.input_spin.setRange(320, 640)
        self.input_spin.setSingleStep(32)
        self.input_spin.setValue(416)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 0.9)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.35)
        self.nms_spin = QDoubleSpinBox()
        self.nms_spin.setRange(0.1, 0.9)
        self.nms_spin.setSingleStep(0.05)
        self.nms_spin.setValue(0.45)
        self.start_btn = QPushButton("开始检测")
        self.stop_btn = QPushButton("停止检测")
        # Landmarks controls
        self.enable_landmarks = QCheckBox("绘制手指关键点(21)")
        self.enable_landmarks.setChecked(True if HAS_MEDIAPIPE else False)
        self.max_hands_spin = QSpinBox()
        self.max_hands_spin.setRange(1, 2)
        self.max_hands_spin.setValue(2)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 0.8)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.15)
        self.only_match_checkbox = QCheckBox("仅与YOLO匹配后绘制关键点")
        self.only_match_checkbox.setChecked(False)
        # 左右手镜像选项
        self.swap_handedness_checkbox = QCheckBox("调换左右手")
        self.swap_handedness_checkbox.setChecked(True)

        # 控制面板布局（恢复 v 与 ctrl_box 的定义）
        ctrl_box = QGroupBox("控制面板")
        v = QVBoxLayout()
        v.addWidget(QLabel("模型选择"))
        v.addWidget(self.model_combo)
        v.addWidget(QLabel("YOLO输入尺寸"))
        v.addWidget(self.input_spin)
        v.addWidget(QLabel("置信度阈值"))
        v.addWidget(self.conf_spin)
        v.addWidget(QLabel("NMS阈值"))
        v.addWidget(self.nms_spin)
        v.addWidget(self.enable_landmarks)
        v.addWidget(QLabel("最大手数"))
        v.addWidget(self.max_hands_spin)
        v.addWidget(QLabel("关键点匹配IOU阈值"))
        v.addWidget(self.iou_spin)
        v.addWidget(self.only_match_checkbox)
        v.addWidget(self.swap_handedness_checkbox)

        hbtn = QHBoxLayout()
        hbtn.addWidget(self.start_btn)
        hbtn.addWidget(self.stop_btn)
        v.addLayout(hbtn)
        ctrl_box.setLayout(v)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.addWidget(ctrl_box)
        # 右侧：视频 + 日志
        right_container = QWidget()
        right_box = QHBoxLayout(right_container)
        right_box.addWidget(self.video_label, 1)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumWidth(280)
        self.log_text.setStyleSheet("background:#111;color:#ddd;")
        log_group = QGroupBox("日志")
        log_v = QVBoxLayout()
        log_v.addWidget(self.log_text)
        log_group.setLayout(log_v)
        right_box.addWidget(log_group)
        layout.addWidget(right_container, 1)
        self.setCentralWidget(root)

        # State
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_frame)
        self.net = None
        self.out_names = []
        self.class_names = ["hand"]
        self.last_detect_count = None
        # 手掌状态（OPEN/CLENCHED）与持续时长统计（分别记录左右手）
        self.hand_states = {"Left": None, "Right": None}
        self.hand_since = {"Left": None, "Right": None}

        # Init models
        cfg_path, weights_path, class_names = ensure_models()
        if not cfg_path or not weights_path:
            self.statusBar().showMessage("模型准备失败，检查网络或models目录")
        else:
            self.class_names = class_names
            available = []
            for stem in [
                "cross-hands-yolov4-tiny",
                "cross-hands-tiny-prn",
                "cross-hands-tiny",
                "cross-hands",
            ]:
                c = MODELS_DIR / f"{stem}.cfg"
                w = MODELS_DIR / f"{stem}.weights"
                if c.exists() and w.exists():
                    available.append((stem, c, w))
            for name, c, w in available:
                self.model_combo.addItem(name, (c, w))
            self.model_combo.setCurrentIndex(0)
            self.load_selected_model()

        # Signals
        self.model_combo.currentIndexChanged.connect(self.load_selected_model)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

    def load_selected_model(self):
        data = self.model_combo.currentData()
        if not data:
            return
        cfg_path, weights_path = data
        try:
            self.net, self.out_names = load_net(cfg_path, weights_path)
            self.statusBar().showMessage(f"使用模型: {cfg_path.name} / {weights_path.name}")
            self.log_event("模型加载", f"{cfg_path.name}/{weights_path.name}")
        except Exception as e:
            self.statusBar().showMessage(f"加载模型失败: {e}")
            self.log_event("模型加载失败", f"{e}")

    def start(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.statusBar().showMessage("无法打开摄像头")
                self.log_event("摄像头打开失败", "无法打开摄像头")
                return
            # init mediapipe hands
            if HAS_MEDIAPIPE and self.enable_landmarks.isChecked():
                try:
                    self.hands = mp_hands.Hands(
                        static_image_mode=False,
                        max_num_hands=int(self.max_hands_spin.value()),
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                except Exception as e:
                    self.hands = None
                    self.statusBar().showMessage(f"初始化关键点失败: {e}")
                    self.log_event("初始化关键点失败", f"{e}")
            self.timer.start(10)
            self.statusBar().showMessage("检测中...")
            self.log_event("开始检测", "OK")
        except Exception as e:
            self.statusBar().showMessage(f"启动失败: {e}")
            self.log_event("启动失败", f"{e}")

    def stop(self):
        self.timer.stop()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self.hands is not None:
            try:
                self.hands.close()
            except Exception:
                pass
            self.hands = None
        self.statusBar().showMessage("已停止")
        self.log_event("停止检测", "OK")

    def log_event(self, name: str, status: str):
         try:
             ts = time.strftime("%H:%M:%S")
             if hasattr(self, "log_text") and self.log_text is not None:
                 self.log_text.append(f"[{ts}] {name} - {status}")
         except Exception:
             pass
 
    def get_extended_count(self, hand_landmarks, W, H) -> int:
        # 简化的张开判定：对食指、中指、无名指、小指，若tip在pip之上（y更小）则认为伸直
        lm = hand_landmarks.landmark
        count = 0
        pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        for tip, pip in pairs:
            try:
                if lm[tip].y < lm[pip].y - 0.02:
                    count += 1
            except Exception:
                pass
        return count

    def update_hand_states(self, per_hand_counts: List[Tuple[str, int]]):
        if not per_hand_counts:
            return
        for label, ext in per_hand_counts:
            if label not in ("Left", "Right"):
                continue
            prev_state = self.hand_states.get(label)
            new_state = None
            if ext >= 3:
                new_state = "OPEN"
            elif ext == 0:
                new_state = "CLENCHED"
            if new_state and new_state != prev_state:
                duration_txt = None
                since = self.hand_since.get(label)
                if prev_state is not None and since is not None:
                    try:
                        duration = time.time() - float(since)
                        duration_txt = f"持续: {duration:.1f}s"
                    except Exception:
                        pass
                self.hand_states[label] = new_state
                self.hand_since[label] = time.time()
                side_cn = "左手" if label == "Left" else "右手"
                if new_state == "OPEN":
                    self.log_event(f"{side_cn}张开", duration_txt or "切换")
                elif new_state == "CLENCHED":
                    self.log_event(f"{side_cn}握紧", duration_txt or "切换")

    def on_frame(self):
        if self.cap is None or self.net is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        conf = float(self.conf_spin.value())
        nms = float(self.nms_spin.value())
        inp = int(self.input_spin.value())
        boxes, confs, cids = detect(frame, self.net, self.out_names, conf_thres=conf, nms_thres=nms, input_size=inp)
        if self.last_detect_count is None or len(boxes) != self.last_detect_count:
            self.log_event("手部检测", f"数量: {len(boxes)}")
            self.last_detect_count = len(boxes)
        vis = draw_detections(frame.copy(), boxes, confs, cids, self.class_names)

        # keypoints
        if self.hands is not None and self.enable_landmarks.isChecked():
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.hands.process(rgb)
                if res.multi_hand_landmarks:
                    W, H = frame.shape[1], frame.shape[0]
                    match_thres = float(self.iou_spin.value())
                    per_hand_counts = []
                    for idx, hand_landmarks in enumerate(res.multi_hand_landmarks):
                        hb = bbox_from_landmarks(hand_landmarks, W, H)
                        draw_this = True
                        if self.only_match_checkbox.isChecked() and boxes:
                            overlaps = [iou(hb, b) for b in boxes]
                            if (not overlaps) or max(overlaps) < match_thres:
                                draw_this = False
                        # 识别左右手
                        label = None
                        try:
                            if hasattr(res, "multi_handedness") and res.multi_handedness and len(res.multi_handedness) > idx:
                                cls = res.multi_handedness[idx].classification[0]
                                label = cls.label  # "Left" 或 "Right"
                                # 根据选项调换左右
                                if self.swap_handedness_checkbox.isChecked():
                                    label = "Left" if label == "Right" else "Right" if label == "Left" else label
                        except Exception:
                            label = None
                        ext = self.get_extended_count(hand_landmarks, W, H)
                        if label in ("Left", "Right"):
                            per_hand_counts.append((label, ext))
                        if draw_this:
                            mp_drawing.draw_landmarks(
                                vis,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_styles.get_default_hand_landmarks_style(),
                                mp_styles.get_default_hand_connections_style(),
                            )
                    self.update_hand_states(per_hand_counts)
            except Exception:
                pass

        # FPS/time
        cv2.putText(vis, time.strftime("%H:%M:%S"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


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


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()