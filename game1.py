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
        # 身体
        pygame.draw.rect(surface, color, pygame.Rect(int(x + w*0.15), int(y + h*0.3), int(w*0.55), int(h*0.5)))
        # 头部
        pygame.draw.circle(surface, color, (int(x + w*0.78), int(y + h*0.35)), int(w*0.18))
        # 眼睛
        pygame.draw.circle(surface, (255, 255, 255), (int(x + w*0.82), int(y + h*0.32)), int(w*0.06))
        pygame.draw.circle(surface, (0, 0, 0), (int(x + w*0.82), int(y + h*0.32)), int(w*0.03))
        # 双腿
        leg_w = int(w*0.18); leg_h = int(h*0.35)
        pygame.draw.rect(surface, color, pygame.Rect(int(x + w*0.20), int(y + h - leg_h), leg_w, leg_h))
        pygame.draw.rect(surface, color, pygame.Rect(int(x + w*0.45), int(y + h - leg_h), leg_w, leg_h))
        # 尾巴
        pts = [(int(x + w*0.10), int(y + h*0.50)), (int(x + w*0.02), int(y + h*0.45)), (int(x + w*0.10), int(y + h*0.70))]
        pygame.draw.polygon(surface, color, pts)

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
        trunk_w = max(6, int(w*0.4))
        trunk = pygame.Rect(int(x + w//2 - trunk_w//2), int(y), trunk_w, h)
        pygame.draw.rect(surface, green, trunk, border_radius=2)
        arm_len = int(h*0.35)
        arm_w = max(6, int(trunk_w*0.8))
        arm_y = int(y + h*0.35)
        left_arm = pygame.Rect(int(x + w//2 - trunk_w//2 - arm_len), arm_y, arm_len, arm_w)
        right_arm = pygame.Rect(int(x + w//2 + trunk_w//2), arm_y, arm_len, arm_w)
        pygame.draw.rect(surface, green, left_arm, border_radius=2)
        pygame.draw.rect(surface, green, right_arm, border_radius=2)
        tip_h = int(h*0.25)
        pygame.draw.rect(surface, green, pygame.Rect(left_arm.left, arm_y - tip_h, arm_w, tip_h), border_radius=2)
        pygame.draw.rect(surface, green, pygame.Rect(right_arm.right - arm_w, arm_y - tip_h, arm_w, tip_h), border_radius=2)

class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("恐龙跳仙人掌 (空格键跳跃)")
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
            now = pygame.time.get_ticks()
            if now - self.last_spawn >= self.next_delta:
                self.spawn()
                self.last_spawn = now
                self.next_delta = random.randint(SPAWN_MIN_MS, SPAWN_MAX_MS)
            self.dino.update()
            for o in list(self.obsts):
                o.update()
                if o.offscreen():
                    self.obsts.remove(o)
            for o in self.obsts:
                if self.dino.rect().colliderect(o.rect()):
                    self.game_over = True
                    break
            if not self.game_over:
                self.score += 1
    def draw(self):
        s = self.screen
        s.fill((255, 255, 255))
        pygame.draw.line(s, (136, 136, 136), (0, GROUND_Y + 40), (WIDTH, GROUND_Y + 40), 1)
        # pygame.draw.rect(s, (51, 51, 51), self.dino.rect())
        self.dino.draw(s)
        for o in self.obsts:
            # pygame.draw.rect(s, (10, 168, 128), o.rect())
            o.draw(s)
        score_text = self.font.render(f"SCORE: {self.score}", True, (0, 0, 0))
        s.blit(score_text, (10, 10))
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
            if self.game_over:
                self.reset()
            else:
                self.dino.jump()
    def run(self):
        while True:
            for e in pygame.event.get():
                self.handle_event(e)
            self.update()
            self.draw()
            self.clock.tick(60)

if __name__ == "__main__":
    try:
        Game().run()
    except Exception as ex:
        print("运行出错:", ex)