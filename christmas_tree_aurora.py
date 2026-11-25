# -*- coding: utf-8 -*-
"""3D 梦幻极光粒子圣诞树 —— pygame + numpy 软件渲染

- 约 1.2 万个发光粒子组成层叠圆锥树冠，正弦抖动形成蓬松下垂的枝簇
- 加法混合 + 多级泛光；青白 / 粉紫 / 冰蓝三段渐变，混入白色闪烁粒子
- 暖金色螺旋灯带、多层径向光晕的树顶金色五角星
- 夜空动态极光（左侧蓝紫 / 右侧青绿）、透视地面星尘、渐变拖尾流星
- 自动缓慢旋转；按住鼠标左键拖动调整视角；ESC 退出
- 全部图形程序化生成，不依赖任何外部图片素材
"""

import math
import os
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

# ------------------------------------------------------------------ 画面参数
WINDOW_W, WINDOW_H = 1280, 800
RENDER_SCALE = 0.60                      # 内部渲染比例（升采样获得柔光）
IW = int(WINDOW_W * RENDER_SCALE) // 2 * 2
IH = int(WINDOW_H * RENDER_SCALE) // 2 * 2
CX, CY = IW / 2.0, IH * 0.535
FOCAL = IH * 1.52
CAM_DIST = 4.65
CAM_Y = 0.08

TREE_BASE = -1.20
TREE_H = 2.38
R_MAX = 1.02
TOP_Y = TREE_BASE + TREE_H

TONE = 168.0                             # 线性曝光
TWO_PI = math.pi * 2.0


def hex_rgb(text):
    text = text.lstrip("#")
    return np.array([int(text[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


C_ICE = hex_rgb("7dd3fc")      # 冰蓝
C_PURPLE = hex_rgb("c084fc")   # 紫
C_PINK = hex_rgb("f0abfc")     # 粉紫
C_TOP = hex_rgb("ccfbf1")      # 青白
C_GOLD = hex_rgb("fde68a")     # 暖金
C_WHITE = np.array([1.0, 0.98, 0.94], np.float32)
C_SKY = np.array([0.62, 0.76, 1.0], np.float32)
C_GROUND = np.array([0.55, 0.80, 1.0], np.float32)


# ------------------------------------------------------------------ 数学工具
def _box(a, radius, axis):
    """沿单轴一次盒式模糊（累积和实现，边缘复制）。"""
    if radius <= 0:
        return a
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (radius + 1, radius)
    p = np.pad(a, pad, mode="edge")
    cs = np.cumsum(p, axis=axis, dtype=np.float32)
    hi = [slice(None)] * a.ndim
    lo = [slice(None)] * a.ndim
    hi[axis] = slice(2 * radius + 1, 2 * radius + 1 + n)
    lo[axis] = slice(0, n)
    return (cs[tuple(hi)] - cs[tuple(lo)]) / float(2 * radius + 1)


def blur(a, radius=2, passes=2):
    for _ in range(passes):
        a = _box(a, radius, 0)
        a = _box(a, radius, 1)
    return a


def _lerp(c0, c1, u):
    return c0 + (c1 - c0) * u


class Meteor:
    """屏幕空间流星，带渐变拖尾。"""

    def __init__(self, rng):
        self.x = float(rng.uniform(IW * 0.03, IW * 0.97))
        self.y = float(rng.uniform(-IH * 0.06, IH * 0.30))
        angle = float(rng.uniform(0.55, 0.95))          # 与水平方向的夹角
        direction = 1.0 if rng.random() < 0.5 else -1.0
        speed = float(rng.uniform(430.0, 760.0))
        self.vx = direction * math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.dur = float(rng.uniform(0.85, 1.45))
        self.life = 0.0
        self.alpha = 0.0

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life += dt
        fade_in = min(1.0, self.life / 0.12)
        fade_out = min(1.0, max(0.0, (self.dur - self.life) / 0.35))
        self.alpha = fade_in * fade_out

    def alive(self):
        return self.life < self.dur and -160 < self.y < IH + 160 and -200 < self.x < IW + 200


# ------------------------------------------------------------------ 场景
class ChristmasScene:
    def __init__(self, screen, seed=20251225):
        self.screen = screen
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.yaw = 0.35
        self.pitch = 0.06
        self.fling = 0.0
        self.dragging = False

        # 累积缓冲采用 (H, W, 3) 行主序布局，输出时可直接交给 frombuffer
        self.acc = np.zeros((IH, IW, 3), np.float32)

        self.meteors = []
        self.meteor_timer = 1.2
        self.aurora_tick = 0

        self._build_tree()
        self._build_helix()
        self._build_ground()
        self._build_sky()
        self._build_star_sprite()
        self._build_background()
        self._build_aurora_grid()
        self.aurora_surf = pygame.Surface((IW, IH))
        self._update_aurora()

        self.font = self._load_font(16)
        self.hint = "按住鼠标左键拖动  调整视角    ·    ESC 退出"
        self.hint_surf = (self.font.render(self.hint, True, (178, 190, 220))
                          if self.font is not None else None)

    # -------------------------------------------------------------- 数据构建
    def _tree_gradient(self, s):
        s = np.clip(s, 0.0, 1.0)[:, None]
        a = np.clip(s / 0.42, 0, 1)
        b = np.clip((s - 0.42) / 0.30, 0, 1)
        c = np.clip((s - 0.72) / 0.28, 0, 1)
        seg1 = _lerp(C_ICE, C_PURPLE, a)
        seg2 = _lerp(C_PURPLE, C_PINK, b)
        seg3 = _lerp(C_PINK, C_TOP, c)
        return np.where(s < 0.42, seg1, np.where(s < 0.72, seg2, seg3))

    def _build_tree(self):
        rng = self.rng
        tiers = 12
        branch, puff, rim = [], [], []

        for j in range(tiers):
            r0 = j / tiers
            r1 = (j + 1.0) / tiers
            env = R_MAX * (1.0 - r0) ** 1.08
            wobble = (1.0 + 0.055 * math.sin(j * 1.9 + 0.6)
                      + 0.028 * math.cos(j * 3.7))
            y_bottom = TREE_BASE + r0 * TREE_H
            y_top = TREE_BASE + r1 * TREE_H
            n_br = max(7, int(round(5 + 16.0 * env / R_MAX)))

            for _ in range(n_br):
                ang0 = rng.random() * TWO_PI + j * 2.399 + rng.normal(0.0, 0.07)
                tip_r = env * wobble * rng.uniform(0.86, 1.04)
                droop = rng.uniform(0.085, 0.170) * (0.45 + 0.55 * env / R_MAX)
                n = int(round((32 + 30 * env / R_MAX) * rng.uniform(0.85, 1.15)))
                if env < R_MAX * 0.35:
                    n = int(round(n * 1.25))

                # 主枝：从内侧上端向外下方伸出，末端加重下垂
                s = rng.random(n) ** 0.72
                r = tip_r * (0.06 + 0.94 * s) + rng.normal(0.0, 0.009, n)
                r += rng.normal(0.0, 0.020, n) * s ** 2
                y = (y_top + (y_bottom - y_top) * 0.62 * s
                     - droop * s ** 2.3
                     + rng.normal(0.0, 0.006 + 0.012 * s, n))
                a = (ang0 + rng.normal(0.0, 0.014, n)
                     + rng.normal(0.0, 0.030, n) * s)
                branch.append(np.stack([r * np.cos(a), y, r * np.sin(a)], 1))

                # 枝端蓬松小簇，制造一簇簇的毛茸感
                for _ in range(2):
                    sc = rng.uniform(0.62, 0.99)
                    cr = tip_r * (0.06 + 0.94 * sc)
                    cy = (y_top + (y_bottom - y_top) * 0.62 * sc
                          - droop * sc ** 2.3)
                    m = int(rng.integers(3, 6))
                    pr = cr + rng.normal(0.0, 0.017, m)
                    py = cy + rng.normal(0.0, 0.015, m)
                    pa = ang0 + rng.normal(0.0, 0.075, m)
                    puff.append(np.stack([pr * np.cos(pa), py, pr * np.sin(pa)], 1))

            # 层缘亮环：勾出每层轮廓，形成层叠剪影
            n_rim = 80
            a = rng.random(n_rim) * TWO_PI
            rr = env * wobble * rng.uniform(0.90, 1.03, n_rim)
            yy = (y_bottom - rng.uniform(0.020, 0.070, n_rim)
                  + 0.010 * np.sin(a * 7.0 + j * 1.7))
            rim.append(np.stack([rr * np.cos(a), yy, rr * np.sin(a)], 1))

        # 体积填充：少量内部星光，增加纵深
        n_fill = 1250
        tt = 1.0 - np.sqrt(rng.random(n_fill))
        rad = (R_MAX * (1.0 - tt) ** 1.08) * (rng.random(n_fill) ** 0.72) * 0.94
        ang = rng.random(n_fill) * TWO_PI
        fill = np.stack([
            rad * np.cos(ang),
            TREE_BASE + tt * TREE_H + rng.normal(0, 0.024, n_fill),
            rad * np.sin(ang),
        ], axis=1)

        b_arr = np.concatenate(branch, axis=0)
        p_arr = np.concatenate(puff, axis=0)
        r_arr = np.concatenate(rim, axis=0)
        pos = np.concatenate([b_arr, p_arr, r_arr, fill], axis=0).astype(np.float32)

        count = pos.shape[0]
        n_rim_start = b_arr.shape[0] + p_arr.shape[0]
        t01 = np.clip((pos[:, 1] - TREE_BASE) / TREE_H, 0.0, 1.0)
        col = self._tree_gradient(t01)

        bright = rng.uniform(0.68, 1.38, count).astype(np.float32)
        rim_slice = slice(n_rim_start, n_rim_start + r_arr.shape[0])
        col[rim_slice] = np.clip(col[rim_slice] * 0.78 + C_WHITE * 0.22, 0.0, 1.0)
        bright[rim_slice] *= 1.38

        spark = rng.random(count) < 0.045
        col[spark] = col[spark] * 0.28 + C_WHITE * 0.90
        bright[spark] = np.maximum(bright[spark], 1.30)

        self.tree_pos = pos
        self.tree_col = col.astype(np.float32)
        self.tree_spark = spark
        self.tree_phase = (rng.random(count) * TWO_PI).astype(np.float32)
        self.tree_freq = rng.uniform(0.8, 3.1, count).astype(np.float32)
        self.tree_bright = bright

    def _build_helix(self):
        rng = self.rng
        m = 1050
        t = np.linspace(0.015, 0.985, m)
        ang = TWO_PI * 5.6 * t + 0.5
        env = R_MAX * (1.0 - t) ** 1.08
        rad = env * 0.965 + 0.012
        y = TREE_BASE + t * TREE_H * 0.975
        self.helix_pos = np.stack([
            rad * np.cos(ang), y, rad * np.sin(ang),
        ], axis=1).astype(np.float32)
        bead = (np.arange(m) % 20 == 0).astype(np.float32)
        self.helix_bright = ((1.25 + 2.10 * bead) * rng.uniform(0.85, 1.15, m)).astype(np.float32)
        self.helix_phase = (t * 26.0 + rng.random(m) * 0.8).astype(np.float32)
        self.helix_freq = rng.uniform(2.0, 3.6, m).astype(np.float32)
        self.helix_col = np.tile(C_GOLD, (m, 1)).astype(np.float32)
        self.helix_col += rng.random((m, 1)).astype(np.float32) * 0.10

    def _build_ground(self):
        rng = self.rng
        n = 1700
        rad = np.sqrt(rng.random(n)) * 3.6
        ang = rng.random(n) * TWO_PI
        y = TREE_BASE - 0.02 + rng.normal(0, 0.006, n)
        self.ground_pos = np.stack([
            rad * np.cos(ang), y, rad * np.sin(ang),
        ], axis=1).astype(np.float32)
        fade = 0.14 + 0.62 * np.exp(-(rad / 2.55) ** 2)
        self.ground_bright = (fade * rng.uniform(0.45, 1.25, n)).astype(np.float32)
        self.ground_phase = (rng.random(n) * TWO_PI).astype(np.float32)
        self.ground_freq = rng.uniform(0.5, 2.4, n).astype(np.float32)
        self.ground_col = np.tile(C_GROUND, (n, 1)).astype(np.float32)
        self.ground_col = self.ground_col * 0.7 + C_WHITE * (rng.random((n, 1)) * 0.30).astype(np.float32)

    def _build_sky(self):
        rng = self.rng
        n = 340
        self.sky_x = rng.uniform(0, IW, n).astype(np.float32)
        self.sky_y = (rng.random(n) ** 1.35 * IH * 0.86).astype(np.float32)
        self.sky_bright = rng.uniform(0.10, 0.55, n).astype(np.float32)
        self.sky_phase = (rng.random(n) * TWO_PI).astype(np.float32)
        self.sky_freq = rng.uniform(0.3, 1.6, n).astype(np.float32)
        tint = rng.random((n, 1)).astype(np.float32)
        self.sky_col = (C_SKY[None, :] * (1.0 - tint * 0.45)
                        + C_WHITE[None, :] * (tint * 0.45)).astype(np.float32)

    def _build_star_sprite(self):
        size = 256
        half = (size - 1) / 2.0
        ax = (np.arange(size) - half) / half
        gx, gy = np.meshgrid(ax, ax)
        r = np.hypot(gx, gy)

        halo = (1.00 * np.exp(-(r / 0.085) ** 2)
                + 0.50 * np.exp(-(r / 0.24) ** 2)
                + 0.22 * np.exp(-(r / 0.60) ** 2))

        star_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pts = []
        for k in range(10):
            radius = (0.42 if k % 2 == 0 else 0.165) * half
            a = -math.pi / 2 + k * math.pi / 5
            pts.append((half + radius * math.cos(a), half + radius * math.sin(a)))
        pygame.draw.polygon(star_surf, (255, 255, 255, 255), pts)
        mask = pygame.surfarray.array_alpha(star_surf).T.astype(np.float32) / 255.0
        mask = blur(mask, 2, 2)

        flare = (np.exp(-(gy / 0.008) ** 2) * np.exp(-(np.abs(gx) / 0.52) ** 2)
                 + np.exp(-(gx / 0.008) ** 2) * np.exp(-(np.abs(gy) / 0.52) ** 2))

        inten = halo * 0.85 + mask * 1.15 + flare * 0.55
        star_col = np.array([1.0, 0.84, 0.46], np.float32)      # 暖金
        sprite = inten[..., None] * star_col
        hot = np.exp(-(r / 0.032) ** 2) * 1.10
        sprite = sprite + hot[..., None] * np.array([1.0, 0.96, 0.86], np.float32)
        sprite = np.clip(sprite, 0.0, 1.0)

        self.star_glow_surf = pygame.surfarray.make_surface(
            (sprite * 255).astype(np.uint8).transpose(1, 0, 2))
        self.star_pos = np.array([[0.0, TOP_Y + 0.10, 0.0]], np.float32)

    def _build_background(self):
        yy = np.arange(IH, dtype=np.float32)[:, None]           # (IH, 1)
        xx = np.arange(IW, dtype=np.float32)[None, :]           # (1, IW)
        v = (yy / IH)[..., None]                                # (IH, 1, 1)

        top = np.array([0.012, 0.016, 0.040], np.float32)
        bottom = np.array([0.028, 0.024, 0.058], np.float32)
        bg = top[None, None, :] * (1.0 - v) + bottom[None, None, :] * v
        bg = bg + np.zeros((1, IW, 1), np.float32)

        dx = (xx - IW * 0.5) / (IW * 0.5)
        dy = (yy - IH * 0.52) / (IH * 0.5)
        halo = np.exp(-(dx * dx * 1.3 + dy * dy * 1.6) * 2.2)
        bg += halo[..., None] * np.array([0.045, 0.030, 0.080], np.float32)

        # 暗角直接烘焙进背景（粒子层不需要再做全屏乘法）
        r2 = dx * dx * 0.85 + dy * dy
        bg *= (1.0 - 0.32 * np.clip(r2, 0, 1.8) ** 1.35)[..., None]
        self.bg = np.ascontiguousarray(bg * TONE, dtype=np.float32)

    def _build_aurora_grid(self):
        gw, gh = 200, 126
        self.aur_x = np.linspace(0.0, 1.0, gw, dtype=np.float32)[None, :]
        self.aur_y = np.linspace(0.0, 1.0, gh, dtype=np.float32)[:, None]
        self.aur_col_l = np.array([0.34, 0.22, 0.95], np.float32)
        self.aur_col_r = np.array([0.10, 0.86, 0.68], np.float32)
        self.aur_col_m = np.array([0.42, 0.30, 0.85], np.float32)

    def _update_aurora(self):
        x, y = self.aur_x, self.aur_y
        t = self.t

        # 左侧蓝紫极光：竖向光帘 + 底部羽毛状波动
        s1 = 0.5 + 0.5 * np.sin(x * 24.0 + t * 0.85 + np.sin(x * 6.5 - t * 0.7) * 2.3)
        s2 = 0.5 + 0.5 * np.sin(x * 43.0 - t * 1.55 + 1.7)
        streak_l = (0.20 + 0.80 * s1 ** 2) * (0.40 + 0.60 * s2)
        edge_l = 0.42 + 0.13 * np.sin(x * 6.0 + t * 0.55) + 0.06 * np.sin(x * 15.0 - t * 1.05)
        v_l = np.clip(1.0 - y / edge_l, 0.0, 1.0) ** 0.92
        mask_l = np.exp(-((x - 0.20) / 0.27) ** 2)
        left = v_l * streak_l * mask_l

        # 右侧青绿极光
        s3 = 0.5 + 0.5 * np.sin(x * 21.0 - t * 0.95 + 2.4 + np.sin(x * 8.0 + t * 0.6) * 2.0)
        s4 = 0.5 + 0.5 * np.sin(x * 37.0 + t * 1.35)
        streak_r = (0.18 + 0.82 * s3 ** 2) * (0.38 + 0.62 * s4)
        edge_r = 0.38 + 0.13 * np.sin(x * 7.0 - t * 0.65 + 1.3) + 0.06 * np.sin(x * 13.0 + t * 1.15)
        v_r = np.clip(1.0 - y / edge_r, 0.0, 1.0) ** 0.92
        mask_r = np.exp(-((x - 0.80) / 0.29) ** 2)
        right = v_r * streak_r * mask_r

        # 中间淡紫过渡
        mid = np.exp(-((x - 0.5) / 0.30) ** 2) * np.exp(-(y / 0.22) ** 2) * 0.30

        # 细竖纹，强化极光光帘质感
        striate = 0.74 + 0.26 * (0.5 + 0.5 * np.sin(x * 118.0 + t * 2.6))

        field = (left[..., None] * self.aur_col_l
                 + right[..., None] * self.aur_col_r
                 + mid[..., None] * self.aur_col_m)
        field *= 0.92 * striate[..., None]
        buf = (np.clip(field, 0.0, 1.0) * 255).astype(np.uint8).transpose(1, 0, 2)
        small = pygame.surfarray.make_surface(buf)
        self.aurora_surf = pygame.transform.smoothscale(small, (IW, IH)).convert()

    @staticmethod
    def _load_font(size):
        for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc",
                     "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"):
            if os.path.exists(path):
                try:
                    return pygame.font.Font(path, size)
                except Exception:
                    pass
        for name in ("microsoftyahei", "pingfangsc", "notosanscjksc", "simhei"):
            try:
                font = pygame.font.SysFont(name, size)
                if font is not None:
                    return font
            except Exception:
                continue
        return pygame.font.Font(None, size)

    # -------------------------------------------------------------- 投影与绘制
    def _project(self, pts, yaw, pitch):
        ca, sa = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        x1 = x * ca + z * sa
        z1 = -x * sa + z * ca
        y1 = y * cp - z1 * sp
        z2 = y * sp + z1 * cp

        zv = np.maximum(CAM_DIST + z2, 0.25)
        scale = FOCAL / zv
        sx = CX + x1 * scale
        sy = CY - (y1 - CAM_Y) * scale
        return sx, sy, zv, scale

    def _splat(self, xs, ys, colors, weights):
        """双线性泼溅，加法累积到 (H, W, 3) 行主序缓冲。"""
        m = ((xs >= 0) & (xs < IW - 1) & (ys >= 0) & (ys < IH - 1)
             & (weights > 1e-4))
        if not m.any():
            return
        xs = xs[m]
        ys = ys[m]
        colors = colors[m]
        weights = weights[m]

        x0 = xs.astype(np.int32)
        y0 = ys.astype(np.int32)
        fx = (xs - x0).astype(np.float32)
        fy = (ys - y0).astype(np.float32)
        base = (y0 * IW + x0) * 3

        w00 = (1.0 - fx) * (1.0 - fy) * weights
        w10 = fx * (1.0 - fy) * weights
        w01 = (1.0 - fx) * fy * weights
        w11 = fx * fy * weights
        step = IW * 3
        idx = np.concatenate([base, base + 3, base + step, base + step + 3])
        w = np.concatenate([w00, w10, w01, w11]).astype(np.float32, copy=False)
        w *= TONE
        colors = np.tile(colors, (4, 1))

        flat = self.acc.reshape(-1)
        for c in range(3):
            np.add.at(flat, idx + c, w * colors[:, c])

    def _emit_tree(self):
        sx, sy, zv, _ = self._project(self.tree_pos, self.yaw, self.pitch)
        osc = 0.5 + 0.5 * np.sin(self.t * self.tree_freq + self.tree_phase)
        w = self.tree_bright * (0.58 + 0.42 * osc)
        if self.tree_spark.any():
            w = np.where(self.tree_spark,
                         self.tree_bright * (0.18 + 2.0 * osc ** 5),
                         w)
        w *= np.clip((CAM_DIST / zv) ** 0.6, 0.50, 1.45)
        self._splat(sx, sy, self.tree_col, w)

    def _emit_helix(self):
        sx, sy, zv, _ = self._project(self.helix_pos, self.yaw, self.pitch)
        osc = 0.5 + 0.5 * np.sin(self.t * self.helix_freq + self.helix_phase)
        w = self.helix_bright * (0.55 + 0.45 * osc ** 1.5)
        w *= np.clip((CAM_DIST / zv) ** 0.6, 0.55, 1.4)
        self._splat(sx, sy, self.helix_col, w)

    def _emit_ground(self):
        pos = self.ground_pos.copy()
        pos[:, 1] += 0.012 * np.sin(self.t * 0.9 + self.ground_phase)
        sx, sy, zv, _ = self._project(pos, self.yaw, self.pitch)
        osc = 0.5 + 0.5 * np.sin(self.t * self.ground_freq + self.ground_phase * 1.7)
        w = self.ground_bright * (0.35 + 0.65 * osc ** 2)
        w *= np.clip((CAM_DIST / zv) ** 0.5, 0.45, 1.5)
        self._splat(sx, sy, self.ground_col, w)

    def _emit_sky(self):
        osc = 0.5 + 0.5 * np.sin(self.t * self.sky_freq + self.sky_phase)
        w = self.sky_bright * (0.20 + 0.80 * osc ** 2)
        self._splat(self.sky_x, self.sky_y, self.sky_col, w)

    def _emit_star(self):
        sx, sy, zv, _ = self._project(self.star_pos, self.yaw, self.pitch)
        size = int(np.clip(122.0 * (CAM_DIST / zv[0]), 78, 168))
        sprite = pygame.transform.smoothscale(self.star_glow_surf, (size, size))
        arr = pygame.surfarray.array3d(sprite).transpose(1, 0, 2).astype(np.float32) / 255.0
        arr *= 1.35 * (0.90 + 0.10 * math.sin(self.t * 2.2)) * TONE

        x0 = int(round(sx[0] - size / 2.0))
        y0 = int(round(sy[0] - size / 2.0))
        dst_x0, dst_y0 = max(0, x0), max(0, y0)
        dst_x1, dst_y1 = min(IW, x0 + size), min(IH, y0 + size)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return
        src = arr[dst_y0 - y0:dst_y1 - y0, dst_x0 - x0:dst_x1 - x0, :]
        self.acc[dst_y0:dst_y1, dst_x0:dst_x1, :] += src

    def _emit_meteors(self):
        head = np.array([0.88, 0.96, 1.0], np.float32)
        tail = np.array([0.55, 0.42, 1.0], np.float32)
        n = 30
        f = np.linspace(0.0, 1.0, n, dtype=np.float32)
        for m in self.meteors:
            if m.alpha <= 0.01:
                continue
            xs = m.x - m.vx * f * 0.20
            ys = m.y - m.vy * f * 0.20
            col = head[None, :] * (1.0 - f[:, None]) + tail[None, :] * f[:, None]
            w = (1.0 - f) ** 2.2 * 1.3 * m.alpha
            self._splat(xs, ys, col, w)
            # 亮头
            self._splat(np.array([m.x], np.float32), np.array([m.y], np.float32),
                        head[None, :], np.array([2.6 * m.alpha], np.float32))

    # -------------------------------------------------------------- 主循环支持
    def step(self, dt):
        self.t += dt
        if not self.dragging:
            self.yaw += self.fling * min(dt * 60.0, 2.0)
            self.fling *= 0.90 ** (dt * 60.0)
        self.yaw += 0.11 * dt

        for m in self.meteors:
            m.update(dt)
        self.meteors = [m for m in self.meteors if m.alive()]

        self.meteor_timer -= dt
        if self.meteor_timer <= 0.0 and len(self.meteors) < 4:
            self.meteors.append(Meteor(self.rng))
            self.meteor_timer = float(self.rng.uniform(2.2, 5.5))

        self.aurora_tick += 1
        if self.aurora_tick % 4 == 0:
            self._update_aurora()

    def _glow_surface(self):
        """低分辨率泛光层：抽样 + 模糊后升到内部尺寸，由主图统一升采样。"""
        small = self.acc[::5, ::5, :].copy()
        small = blur(small, 3, 1)
        small *= 165.0 / TONE
        buf = np.clip(small, 0, 255).astype(np.uint8)
        surf = pygame.image.frombuffer(buf.tobytes(), (buf.shape[1], buf.shape[0]), "RGB")
        return pygame.transform.smoothscale(surf, (IW, IH))

    def draw(self):
        self.acc[:] = 0.0
        self._emit_sky()
        self._emit_ground()
        self._emit_tree()
        self._emit_helix()
        self._emit_star()
        self._emit_meteors()

        glow = self._glow_surface()

        img = self.acc + self.bg
        np.minimum(img, 255.0, out=img)
        frame = img.astype(np.uint8)
        surf = pygame.image.frombuffer(frame.tobytes(), (IW, IH), "RGB").convert()

        # 泛光与极光在内部尺寸叠加，只做一次升采样
        surf.blit(glow, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        surf.blit(self.aurora_surf, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

        scaled = pygame.transform.smoothscale(surf, (WINDOW_W, WINDOW_H))
        self.screen.blit(scaled, (0, 0))

        if self.hint_surf is not None:
            alpha = int(150 * min(1.0, max(0.0, (self.t - 0.6) / 1.4)))
            if alpha > 0:
                self.hint_surf.set_alpha(alpha)
                self.screen.blit(self.hint_surf, (26, WINDOW_H - 36))


def main():
    pygame.init()
    pygame.display.set_caption("3D 梦幻极光粒子圣诞树")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()
    scene = ChristmasScene(screen)

    dragging = False
    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
                scene.dragging = True
                scene.fling = 0.0
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
                scene.dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx, dy = event.rel
                scene.yaw += dx * 0.006
                scene.pitch = max(-0.22, min(0.42, scene.pitch + dy * 0.004))
                scene.fling = dx * 0.006

        scene.step(dt)
        scene.draw()
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
