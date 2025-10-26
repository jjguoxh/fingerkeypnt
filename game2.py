import sys
import math
import random
import pygame
import os
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

WIDTH, HEIGHT = 960, 540
GROUND_Y = HEIGHT - 80
GRAVITY = 0.6
MAX_PULL = 140
POWER_SCALE = 0.25
FRICTION = 0.995
STOP_SPEED = 0.35

SLING_POS = pygame.Vector2(150, GROUND_Y - 60)
BIRD_RADIUS = 12
PIG_RADIUS = 16
BLOCK_SIZE = (40, 20)

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

class PinchDetector:
    def __init__(self):
        self.cap = None
        self.hands = None
        self.ok = False
        self.last_pinch = False
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
    def poll_pinch(self):
        # 返回 (是否捏合, 归一化X, 归一化Y)
        if not self.ok or self.cap is None or self.hands is None:
            return False, None, None
        ret, frame = self.cap.read()
        if not ret:
            return False, None, None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)
            is_pinch = False
            nx = ny = None
            if res.multi_hand_landmarks:
                # 取第一只手
                lm = res.multi_hand_landmarks[0].landmark
                dx = lm[8].x - lm[4].x
                dy = lm[8].y - lm[4].y
                dist = (dx * dx + dy * dy) ** 0.5
                is_pinch = dist < 0.06  # 与 qt_app.py 阈值一致
                # 用拇指与食指中点作为拉拽点
                nx = (lm[4].x + lm[8].x) * 0.5
                ny = (lm[4].y + lm[8].y) * 0.5
            self.last_pinch = is_pinch
            return is_pinch, nx, ny
        except Exception:
            return False, None, None
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

class Bird:
    def __init__(self):
        self.pos = SLING_POS.copy()
        self.vel = pygame.Vector2(0, 0)
        self.launched = False
        self.alive = True
        self.color = (220, 50, 50)

    def rect(self):
        return pygame.Rect(self.pos.x - BIRD_RADIUS, self.pos.y - BIRD_RADIUS,
                           BIRD_RADIUS * 2, BIRD_RADIUS * 2)

    def update(self):
        if not self.alive:
            return
        if self.launched:
            self.vel.y += GRAVITY
            self.pos += self.vel
            # 地面碰撞：简单停住与少量摩擦
            if self.pos.y + BIRD_RADIUS >= GROUND_Y:
                self.pos.y = GROUND_Y - BIRD_RADIUS
                self.vel.y *= -0.25
                self.vel.x *= 0.9
                # 速度很小则认为停止
                if self.vel.length() < STOP_SPEED:
                    self.vel.update(0, 0)
                    self.launched = False
            else:
                # 空中摩擦（很小）
                self.vel *= FRICTION

    def draw(self, s):
        pygame.draw.circle(s, self.color, (int(self.pos.x), int(self.pos.y)), BIRD_RADIUS)
        # 简单的眼睛和喙
        eye = (int(self.pos.x + 4), int(self.pos.y - 3))
        pygame.draw.circle(s, (255, 255, 255), eye, 3)
        pygame.draw.circle(s, (0, 0, 0), eye, 1)
        beak = [(self.pos.x + BIRD_RADIUS, self.pos.y),
                (self.pos.x + BIRD_RADIUS + 8, self.pos.y - 3),
                (self.pos.x + BIRD_RADIUS + 8, self.pos.y + 3)]
        pygame.draw.polygon(s, (255, 180, 0), beak)

class Pig:
    def __init__(self, x, y):
        self.pos = pygame.Vector2(x, y)
        self.radius = PIG_RADIUS
        self.alive = True

    def rect(self):
        return pygame.Rect(self.pos.x - self.radius, self.pos.y - self.radius,
                           self.radius * 2, self.radius * 2)

    def draw(self, s):
        if not self.alive:
            return
        pygame.draw.circle(s, (100, 200, 100), (int(self.pos.x), int(self.pos.y)), self.radius)
        pygame.draw.circle(s, (40, 120, 40), (int(self.pos.x), int(self.pos.y)), self.radius, 2)
        # 眼睛
        pygame.draw.circle(s, (255, 255, 255), (int(self.pos.x - 4), int(self.pos.y - 3)), 3)
        pygame.draw.circle(s, (0, 0, 0), (int(self.pos.x - 4), int(self.pos.y - 3)), 1)
        pygame.draw.circle(s, (255, 255, 255), (int(self.pos.x + 4), int(self.pos.y - 3)), 3)
        pygame.draw.circle(s, (0, 0, 0), (int(self.pos.x + 4), int(self.pos.y - 3)), 1)
        # 鼻子
        nose_rect = pygame.Rect(int(self.pos.x - 8), int(self.pos.y + 2), 16, 8)
        pygame.draw.rect(s, (160, 230, 160), nose_rect, border_radius=4)
        pygame.draw.circle(s, (0, 0, 0), (int(self.pos.x - 3), int(self.pos.y + 6)), 1)
        pygame.draw.circle(s, (0, 0, 0), (int(self.pos.x + 3), int(self.pos.y + 6)), 1)

class Block:
    def __init__(self, x, y, w=BLOCK_SIZE[0], h=BLOCK_SIZE[1]):
        self.rect_ = pygame.Rect(x, y, w, h)
        self.alive = True

    def rect(self):
        return self.rect_

    def draw(self, s):
        if not self.alive:
            return
        pygame.draw.rect(s, (120, 120, 120), self.rect_)
        pygame.draw.rect(s, (90, 90, 90), self.rect_, 2)

class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("愤怒的小鸟 (摄像头捏合控制原型)")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        # 英文字体（作为兜底）
        self.font = pygame.font.SysFont("monospace", 18, bold=True)
        self.big_font = pygame.font.SysFont("sans-serif", 24, bold=True)
        # 中文字体：优先 freetype，其次直接 Font
        self.cn_use_freetype = False
        self.cn_font_path = find_cn_font_path()
        if self.cn_font_path:
            try:
                import pygame.freetype as freetype
                self.cn_use_freetype = True
                self.cn_small = freetype.Font(self.cn_font_path, 18)
                self.cn_big = freetype.Font(self.cn_font_path, 24)
            except Exception:
                self.cn_small = pygame.font.Font(self.cn_font_path, 18)
                self.cn_big = pygame.font.Font(self.cn_font_path, 24)
        else:
            # 若无法找到中文字体，仍使用英文字体（可能无法显示中文）
            self.cn_small = self.font
            self.cn_big = self.big_font
        self.pinch = PinchDetector()
        self.reset()

    def reset(self):
        self.score = 0
        self.level_time = 0
        self.dragging = False
        self.drag_pos = SLING_POS.copy()
        self.birds_remaining = 3
        self.current_bird = Bird()
        self.pigs = [Pig(WIDTH - 180, GROUND_Y - 24), Pig(WIDTH - 120, GROUND_Y - 24)]
        # 简单木块/石块结构
        self.blocks = []
        base_y = GROUND_Y - BLOCK_SIZE[1]
        for i in range(4):
            self.blocks.append(Block(WIDTH - 260 + i * 44, base_y))
        self.blocks.append(Block(WIDTH - 240, base_y - 22))
        self.blocks.append(Block(WIDTH - 200, base_y - 22))

    def sling_vector(self):
        # 从弹弓到拖拽点的向量（指向弹弓被拉的方向）
        v = self.drag_pos - SLING_POS
        length = v.length()
        if length > MAX_PULL:
            v.scale_to_length(MAX_PULL)
            self.drag_pos = SLING_POS + v
        return v

    def launch(self):
        if self.current_bird.launched or self.birds_remaining <= 0:
            return
        v = self.sling_vector()
        # 发射方向与速度与拉伸长度相反
        self.current_bird.vel = -v * POWER_SCALE
        self.current_bird.launched = True
        self.birds_remaining -= 1

    def update(self):
        self.level_time += self.clock.get_time()
        # 摄像头捏合控制弹弓：捏合=拖拽，松开=发射
        try:
            if hasattr(self, "pinch") and self.pinch and self.pinch.ok:
                is_pinch, nx, ny = self.pinch.poll_pinch()
                if is_pinch and nx is not None and ny is not None and not self.current_bird.launched:
                    self.dragging = True
                    pinch_pos = pygame.Vector2(nx * WIDTH, ny * HEIGHT)
                    v = pinch_pos - SLING_POS
                    if v.length() > MAX_PULL:
                        v.scale_to_length(MAX_PULL)
                    self.drag_pos = SLING_POS + v
                    self.current_bird.pos = SLING_POS + v
                elif (not is_pinch) and self.dragging:
                    self.dragging = False
                    self.launch()
        except Exception:
            pass
        self.current_bird.update()
        # 与地面无额外处理，已在 bird.update 中
        # 与方块/猪的简单碰撞：圆与矩形、圆与圆
        if self.current_bird.alive and (self.current_bird.vel.length() > 0.01 or self.current_bird.launched):
            bird_rect = self.current_bird.rect()
            # 方块
            for b in self.blocks:
                if not b.alive:
                    continue
                if bird_rect.colliderect(b.rect()):
                    b.alive = False
                    # 轻微反弹与减速
                    self.current_bird.vel *= 0.6
                    self.score += 10
            # 猪
            for p in self.pigs:
                if not p.alive:
                    continue
                d = (self.current_bird.pos - p.pos).length()
                if d <= (BIRD_RADIUS + p.radius):
                    p.alive = False
                    self.current_bird.vel *= 0.5
                    self.score += 50
        # 当前鸟停止且仍有鸟时，允许空格生成下一只
        if not self.current_bird.launched and self.current_bird.vel.length() == 0:
            pass

    def draw(self):
        s = self.screen
        s.fill((200, 220, 255))
        # 地面
        pygame.draw.rect(s, (150, 120, 90), (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))
        pygame.draw.line(s, (120, 90, 60), (0, GROUND_Y), (WIDTH, GROUND_Y), 2)
        # 弹弓支架
        pygame.draw.line(s, (90, 70, 50), SLING_POS, (SLING_POS.x - 12, SLING_POS.y + 40), 6)
        pygame.draw.line(s, (90, 70, 50), SLING_POS, (SLING_POS.x + 12, SLING_POS.y + 40), 6)
        # 弹弓皮筋（拖拽时显示）
        if self.dragging:
            v = self.sling_vector()
            tip = self.drag_pos
            pygame.draw.line(s, (40, 30, 20), (SLING_POS.x - 8, SLING_POS.y), tip, 3)
            pygame.draw.line(s, (40, 30, 20), (SLING_POS.x + 8, SLING_POS.y), tip, 3)
        # 方块与猪
        for b in self.blocks:
            b.draw(s)
        for p in self.pigs:
            p.draw(s)
        # 小鸟
        self.current_bird.draw(s)
        # UI 文本（使用中文字体）
        if self.cn_use_freetype:
            self.cn_small.render_to(s, (10, 10), f"得分: {self.score}", (20, 20, 20))
            self.cn_small.render_to(s, (10, 34), f"剩余小鸟: {self.birds_remaining}", (20, 20, 20))
            instr = [
                "摄像头：拇指与食指捏合拖拽，松开发射",
                "空格：如果当前小鸟已停止，生成下一只",
                "R：重置关卡",
            ]
            for i, t in enumerate(instr):
                self.cn_small.render_to(s, (10, 60 + i * 22), t, (20, 20, 20))
        else:
            info1 = self.cn_small.render(f"得分: {self.score}", True, (20, 20, 20))
            s.blit(info1, (10, 10))
            info2 = self.cn_small.render(f"剩余小鸟: {self.birds_remaining}", True, (20, 20, 20))
            s.blit(info2, (10, 34))
            instr = [
                "摄像头：拇指与食指捏合拖拽，松开发射",
                "空格：如果当前小鸟已停止，生成下一只",
                "R：重置关卡",
            ]
            for i, t in enumerate(instr):
                s.blit(self.cn_small.render(t, True, (20, 20, 20)), (10, 60 + i * 22))

    def handle_event(self, e):
        # 改为摄像头捏合控制，不再处理鼠标拖拽
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_SPACE:
                # 当前鸟停止则补充下一只
                if not self.current_bird.launched and self.current_bird.vel.length() == 0 and self.birds_remaining > 0:
                    self.current_bird = Bird()
            elif e.key == pygame.K_r:
                self.reset()

    def run(self):
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                else:
                    self.handle_event(e)
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)

if __name__ == "__main__":
    g = None
    try:
        g = Game()
        g.run()
    except Exception as ex:
        print("运行出错:", ex)
    finally:
        try:
            if g and hasattr(g, "pinch") and g.pinch:
                g.pinch.close()
        except Exception:
            pass