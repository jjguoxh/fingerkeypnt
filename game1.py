import sys
import os
import random
import pygame

WIDTH, HEIGHT = 800, 200
GROUND_Y = 160
GRAVITY = 0.8
JUMP_VEL = -12
OBST_SPEED = 6
SPAWN_MIN_MS = 1200
SPAWN_MAX_MS = 2200

# 查找支持中文的字体路径
def find_cn_font_path():
    try:
        candidates = [
            "Arial Unicode MS","Microsoft YaHei","SimHei","SimSun",
            "PingFang SC","Hiragino Sans GB","STHeiti","STSong",
            "Songti SC","Heiti SC","Noto Sans SC","Noto Serif SC",
            "Source Han Sans CN","Source Han Serif CN"
        ]
        path = pygame.font.match_font(",".join(candidates))
    except Exception:
        path = None
    if path:
        return path
    local_dir = os.path.join(os.path.dirname(__file__), "fonts")
    dirs = [local_dir, "/System/Library/Fonts","/Library/Fonts","/System/Library/Fonts/Supplemental"]
    keys = ["PingFang","Hiragino","Heiti","Song","YaHei","SimSun","SimHei","NotoSans","Noto Serif","SourceHan","ArialUnicode","Arial Unicode MS"]
    for d in dirs:
        try:
            for name in os.listdir(d):
                lower = name.lower()
                if any(k.lower() in lower for k in keys):
                    p = os.path.join(d, name)
                    if p.lower().endswith((".ttf",".otf",".ttc")):
                        return p
        except Exception:
            pass
    return None

class Dino:
    def __init__(self):
        self.x, self.y = 50, GROUND_Y
        self.w, self.h = 40, 40
        self.vy = 0.0
        self.on_ground = True
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)
    def jump(self):
        if self.on_ground:
            self.on_ground = False
            self.vy = JUMP_VEL
    def update(self):
        self.vy += GRAVITY
        self.y += self.vy
        if self.y >= GROUND_Y:
            self.y = GROUND_Y
            self.vy = 0.0
            self.on_ground = True
    def draw(self, surface):
        r = self.rect()
        x, y, w, h = r.x, r.y, r.w, r.h
        color = (51, 51, 51)
        # 直立身体（窄而高）
        torso_w = int(w * 0.32)
        torso_h = int(h * 0.62)
        torso_x = int(x + w * 0.34)
        torso_y = int(y + h * 0.25)
        pygame.draw.rect(surface, color, pygame.Rect(torso_x, torso_y, torso_w, torso_h))
        # 头部（在身体上方居中偏右）
        head_r = int(w * 0.16)
        head_cx = int(x + w * 0.50)
        head_cy = int(y + h * 0.15)
        pygame.draw.circle(surface, color, (head_cx, head_cy), head_r)
        # 眼睛
        eye_r_white = max(2, int(w * 0.06))
        eye_r_black = max(1, int(w * 0.03))
        eye_cx = int(head_cx + w * 0.05)
        eye_cy = int(head_cy - h * 0.02)
        pygame.draw.circle(surface, (255, 255, 255), (eye_cx, eye_cy), eye_r_white)
        pygame.draw.circle(surface, (0, 0, 0), (eye_cx, eye_cy), eye_r_black)
        # 小短臂（身体右侧）
        arm_w = max(4, int(w * 0.12))
        arm_h = max(4, int(h * 0.06))
        arm_x = int(torso_x + torso_w - arm_w // 2)
        arm_y = int(torso_y + h * 0.18)
        pygame.draw.rect(surface, color, pygame.Rect(arm_x, arm_y, arm_w, arm_h))
        pygame.draw.rect(surface, color, pygame.Rect(arm_x + arm_w // 2, arm_y, max(3, int(w * 0.06)), int(h * 0.14)))
        # 双腿（在底部）
        leg_w = max(6, int(w * 0.10))
        leg_h = max(8, int(h * 0.26))
        left_leg_x = int(x + w * 0.38)
        right_leg_x = int(x + w * 0.56)
        leg_y = int(y + h - leg_h)
        pygame.draw.rect(surface, color, pygame.Rect(left_leg_x, leg_y, leg_w, leg_h))
        pygame.draw.rect(surface, color, pygame.Rect(right_leg_x, leg_y, leg_w, leg_h))
        # 尾巴（左后侧三角形）
        tail_pts = [
            (int(x + w * 0.28), int(y + h * 0.56)),
            (int(x + w * 0.12), int(y + h * 0.50)),
            (int(x + w * 0.26), int(y + h * 0.72)),
        ]
        pygame.draw.polygon(surface, color, tail_pts)
        # 背鳍（两块小三角形）
        spike1 = [
            (int(torso_x), int(torso_y + h * 0.10)),
            (int(torso_x - w * 0.08), int(torso_y + h * 0.08)),
            (int(torso_x), int(torso_y + h * 0.18)),
        ]
        spike2 = [
            (int(torso_x), int(torso_y + h * 0.24)),
            (int(torso_x - w * 0.07), int(torso_y + h * 0.22)),
            (int(torso_x), int(torso_y + h * 0.30)),
        ]
        pygame.draw.polygon(surface, color, spike1)
        pygame.draw.polygon(surface, color, spike2)

class Obstacle:
    def __init__(self):
        h = random.randint(30, 50)
        w = random.randint(20, 30)
        self.w, self.h = w, h
        self.x = WIDTH + 10
        self.y = GROUND_Y + (40 - h)
        self.speed = OBST_SPEED
    def update(self):
        self.x -= self.speed
    def offscreen(self):
        return self.x + self.w < 0
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)
    def draw(self, surface):
        r = self.rect()
        x, y, w, h = r.x, r.y, r.w, r.h
        green = (10, 168, 128)
        # 主干圆角矩形（更像仙人掌）
        trunk_w = max(8, int(w * 0.38))
        trunk = pygame.Rect(int(x + w // 2 - trunk_w // 2), int(y), trunk_w, h)
        pygame.draw.rect(surface, green, trunk, border_radius=4)
        # 两侧手臂（圆角）
        arm_len = int(h * 0.40)
        arm_w = max(6, int(trunk_w * 0.80))
        arm_y = int(y + h * 0.36)
        left_arm = pygame.Rect(int(x + w // 2 - trunk_w // 2 - arm_len), arm_y, arm_len, arm_w)
        right_arm = pygame.Rect(int(x + w // 2 + trunk_w // 2), arm_y, arm_len, arm_w)
        pygame.draw.rect(surface, green, left_arm, border_radius=4)
        pygame.draw.rect(surface, green, right_arm, border_radius=4)
        # 手臂竖枝
        tip_h = int(h * 0.28)
        pygame.draw.rect(surface, green, pygame.Rect(left_arm.left, arm_y - tip_h, arm_w, tip_h), border_radius=4)
        pygame.draw.rect(surface, green, pygame.Rect(right_arm.right - arm_w, arm_y - tip_h, arm_w, tip_h), border_radius=4)
        # 简单刺点（小三角形）
        spike_sz = max(3, int(w * 0.06))
        spikes = [
            (int(trunk.left + 4), int(y + h * 0.20)),
            (int(trunk.left + 6), int(y + h * 0.55)),
            (int(trunk.right - 6), int(y + h * 0.30)),
            (int(trunk.right - 4), int(y + h * 0.70)),
        ]
        for sx, sy in spikes:
            tri = [(sx, sy), (sx - spike_sz, sy + spike_sz), (sx + spike_sz, sy + spike_sz)]
            pygame.draw.polygon(surface, green, tri)

# 新增摄像头与手势识别依赖
import time
try:
    import cv2
except Exception:
    cv2 = None
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    mp = None
    HAS_MEDIAPIPE = False

SPAWN_MAX_MS = 2200

# 基于 qt_app.py 的简化版：统计伸直手指数量
def count_extended(hand_landmarks):
    try:
        lm = hand_landmarks.landmark
        count = 0
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if lm[tip].y < lm[pip].y - 0.02:
                count += 1
        return count
    except Exception:
        return 0

class HandOpenDetector:
    def __init__(self):
        self.cap = None
        self.hands = None
        self.last_clenched = False
        self.last_presence = False
        self.ok = False
        if cv2 is None or not HAS_MEDIAPIPE:
            return
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = None
                return
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.ok = True
        except Exception:
            self.ok = False
    def poll_edge_clenched(self):
        # 返回是否从“非握紧”切换到“握紧”（边沿触发）
        if not self.ok or self.cap is None or self.hands is None:
            self.last_presence = False
            return False
        ret, frame = self.cap.read()
        if not ret:
            self.last_presence = False
            return False
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)
            is_clenched = False
            self.last_presence = bool(getattr(res, "multi_hand_landmarks", None))
            if res.multi_hand_landmarks:
                for hand_landmarks in res.multi_hand_landmarks:
                    ext = count_extended(hand_landmarks)
                    if ext == 0:
                        is_clenched = True
                        break
            edge = (not self.last_clenched) and is_clenched
            self.last_clenched = is_clenched
            return edge
        except Exception:
            self.last_presence = False
            return False
    def has_hand(self):
        return bool(self.last_presence)
    def close(self):
        try:
            if self.hands:
                self.hands.close()
        except Exception:
            pass
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass

class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("恐龙跳仙人掌 (摄像头握紧手掌跳跃)")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16, bold=True)
        self.big_font = pygame.font.SysFont("sans-serif", 22, bold=True)
        # 中文字体：优先 freetype，其次直接 Font
        self.cn_use_freetype = False
        self.cn_font_path = find_cn_font_path()
        if self.cn_font_path:
            try:
                import pygame.freetype as freetype
                self.cn_use_freetype = True
                self.cn_small = freetype.Font(self.cn_font_path, 16)
                self.cn_big = freetype.Font(self.cn_font_path, 22)
            except Exception:
                self.cn_small = pygame.font.Font(self.cn_font_path, 16)
                self.cn_big = pygame.font.Font(self.cn_font_path, 22)
        # 摄像头与手势识别
        self.hand = HandOpenDetector()
        self.paused_for_no_hand = False
        self.reset()
    def reset(self):
        self.dino = Dino()
        self.obsts = []
        self.score = 0
        self.game_over = False
        self.last_spawn = pygame.time.get_ticks()
        self.next_delta = random.randint(SPAWN_MIN_MS, SPAWN_MAX_MS)
    def spawn(self):
        self.obsts.append(Obstacle())
    def update(self):
        if not self.game_over:
            # 摄像头检测：仅当检测到手时卷轴滚动，否则暂停并提示
            try:
                if self.hand:
                    # 更新握紧边沿（跳跃）并同时刷新 presence
                    if self.hand.poll_edge_clenched():
                        self.dino.jump()
                    hand_present = self.hand.has_hand()
                else:
                    hand_present = False
            except Exception:
                hand_present = False

            self.paused_for_no_hand = not hand_present

            now = pygame.time.get_ticks()
            if not self.paused_for_no_hand:
                # 正常生成障碍
                if now - self.last_spawn >= self.next_delta:
                    self.spawn()
                    self.last_spawn = now
                    self.next_delta = random.randint(SPAWN_MIN_MS, SPAWN_MAX_MS)

            # 恐龙自身更新（保持重力与站立逻辑）
            self.dino.update()

            # 卷轴速度：有手则正常移动，无手则速度为0
            cur_speed = OBST_SPEED if not self.paused_for_no_hand else 0
            for o in list(self.obsts):
                o.speed = cur_speed
                o.update()
                if o.offscreen():
                    self.obsts.remove(o)

            # 碰撞检测仍然有效（即使暂停，若已接近则会判定）
            for o in self.obsts:
                if self.dino.rect().colliderect(o.rect()):
                    self.game_over = True
                    break

            # 计分：仅在滚动时增加
            if not self.game_over and not self.paused_for_no_hand:
                self.score += 1
    def draw(self):
        s = self.screen
        s.fill((255, 255, 255))
        pygame.draw.line(s, (136, 136, 136), (0, GROUND_Y + 40), (WIDTH, GROUND_Y + 40), 1)
        # 使用自定义形状绘制恐龙与仙人掌
        self.dino.draw(s)
        for o in self.obsts:
            o.draw(s)
        score_text = self.font.render(f"SCORE: {self.score}", True, (0, 0, 0))
        s.blit(score_text, (10, 10))
        # 未检测到手的提示覆盖层
        if (not self.game_over) and self.paused_for_no_hand:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 120))
            s.blit(overlay, (0, 0))
            msg = "未检测到手，卷轴已暂停"
            if getattr(self, "cn_use_freetype", False) and hasattr(self, "cn_big"):
                rect = self.cn_big.get_rect(msg)
                x = WIDTH // 2 - rect.width // 2
                y = HEIGHT // 2 - rect.height // 2
                self.cn_big.render_to(s, (x, y), msg, (200, 0, 0))
            elif hasattr(self, "cn_big"):
                text = self.cn_big.render(msg, True, (200, 0, 0))
                s.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))
            else:
                text = self.big_font.render(msg, True, (200, 0, 0))
                s.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))
        if self.game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            s.blit(overlay, (0, 0))
            text = "游戏结束 - 空格键重新开始"
            if getattr(self, "cn_use_freetype", False) and hasattr(self, "cn_big"):
                rect = self.cn_big.get_rect(text)
                x = WIDTH // 2 - rect.width // 2
                y = HEIGHT // 2 - rect.height // 2
                self.cn_big.render_to(s, (x, y), text, (255, 255, 255))
            elif hasattr(self, "cn_big"):
                msg = self.cn_big.render(text, True, (255, 255, 255))
                s.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - msg.get_height() // 2))
            else:
                msg = self.big_font.render(text, True, (255, 255, 255))
                s.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - msg.get_height() // 2))
        pygame.display.flip()
    def handle_event(self, e):
        if e.type == pygame.QUIT:
            pygame.quit(); sys.exit(0)
        if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
            # 保留空格用于重新开始，不再用于跳跃
            if self.game_over:
                self.reset()
    def run(self):
        while True:
            for e in pygame.event.get():
                self.handle_event(e)
            self.update()
            self.draw()
            self.clock.tick(60)

if __name__ == "__main__":
    try:
        g = Game()
        g.run()
    except Exception as ex:
        print("运行出错:", ex)
    finally:
        try:
            if hasattr(g, "hand") and g.hand:
                g.hand.close()
        except Exception:
            pass