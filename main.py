"""Music Viewer - a 3D visualizer for whatever your PC is playing.

Run:  python main.py            (add --fullscreen to start fullscreen)
"""
import argparse
import colorsys
import ctypes
import math
import time
from collections import deque

import moderngl
import numpy as np
import pygame

from audio import AudioAnalyzer
from scene import Scene

PALETTES = [  # base hue, hue spread across the bars, saturation
    ("Aurora", 0.62, 0.45, 0.85),
    ("Sunset", 0.95, 0.18, 0.90),
    ("Toxic", 0.28, 0.20, 0.90),
    ("Ice", 0.52, 0.10, 0.70),
    ("Rainbow", 0.00, 1.00, 0.90),
]

HELP = [
    "Drag  orbit camera      Wheel  zoom",
    "F / F11  fullscreen     R  auto-orbit",
    "C  next palette         Up/Down  sensitivity",
    "[ / ]  bloom            H  help      Esc  quit",
]


def hsl(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h % 1.0, l, s)
    return (r, g, b)


class Visualizer:
    def __init__(self, args):
        try:  # crisp rendering on high-DPI displays
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        pygame.display.set_mode((args.width, args.height), flags, vsync=1)
        pygame.display.set_caption("Music Viewer")

        self.ctx = moderngl.create_context()
        self.scene = Scene(self.ctx, pygame.display.get_window_size())
        self.audio = AudioAnalyzer()
        self.font = pygame.font.SysFont("Segoe UI", 17)
        self.font_b = pygame.font.SysFont("Segoe UI Semibold", 20)

        # user settings
        self.palette = 0
        self.sensitivity = 1.0
        self.bloom = 1.0
        self.auto_orbit = True
        self.show_help = False
        self.help_until = self.hud_until = time.perf_counter() + 8.0  # intro

        # camera
        self.yaw, self.pitch, self.dist = 0.0, 0.42, 12.5
        self.dragging = False

        # dynamics
        self.bass = self.mid = self.treble = 0.0
        self.bands = np.zeros(Scene.BAR_BANDS, np.float32)
        self.bass_hist = deque(maxlen=45)
        self.pulse = 0.0
        self.last_beat = -1.0
        self.hue_drift = 0.0
        self.rot = dict(core_ry=0.0, core_rx=0.0, wire_ry=0.0, wire_rz=0.0, pts_ry=0.0, pts_rx=0.0)

        self.fps = 0.0
        self._hud_key = None
        if args.fullscreen:
            pygame.display.toggle_fullscreen()

    # ------------------------------------------------------------ dynamics
    @staticmethod
    def follow(cur, target, dt):
        """Fast attack, slower release."""
        k = np.where(target > cur, 30.0, 7.0)
        return cur + (target - cur) * (1 - np.exp(-dt * k))

    def update(self, dt, t):
        self.audio.analyze(dt)
        s = self.sensitivity
        b, m, tr = self.audio.levels()
        self.bass = float(self.follow(self.bass, min(1.0, b * s * 1.1), dt))
        self.mid = float(self.follow(self.mid, min(1.0, m * s * 1.4), dt))
        self.treble = float(self.follow(self.treble, min(1.0, tr * s * 2.2), dt))

        boost = 1 + np.arange(Scene.BAR_BANDS) / Scene.BAR_BANDS * 1.4  # highs are quieter
        target = np.minimum(1.1, self.audio.bands() * s * 0.85 * boost)
        self.bands = self.follow(self.bands, target, dt).astype(np.float32)

        # beat: bass spike above its recent average
        self.bass_hist.append(self.bass)
        avg = sum(self.bass_hist) / len(self.bass_hist)
        if self.bass > avg * 1.18 and self.bass > 0.35 and t - self.last_beat > 0.28:
            self.pulse, self.last_beat = 1.0, t
            self.hue_drift += 0.025
        self.pulse = max(0.0, self.pulse - dt * 3.5)
        self.hue_drift += dt * 0.01

        r = self.rot
        r["core_ry"] += dt * (0.15 + self.mid * 0.6)
        r["core_rx"] += dt * 0.07
        r["wire_ry"] -= dt * (0.1 + self.treble * 0.8)
        r["wire_rz"] += dt * 0.05
        r["pts_ry"] += dt * (0.02 + self.treble * 0.25)
        r["pts_rx"] += dt * 0.01

        if self.auto_orbit and not self.dragging:
            self.yaw += dt * 0.1

    def frame_state(self, t, hud_alpha):
        name, base, spread, sat = PALETTES[self.palette]
        hue = (base + self.hue_drift * (1.0 if spread > 0.5 else 0.3)) % 1.0
        cp = math.cos(self.pitch)
        eye = (self.dist * cp * math.sin(self.yaw), self.dist * math.sin(self.pitch),
               self.dist * cp * math.cos(self.yaw))
        return dict(
            t=t, eye=eye, fov=60 - self.pulse * 2.5,
            bass=self.bass, mid=self.mid, treble=self.treble, pulse=self.pulse,
            hue=hue, sat=sat, spread=spread, bands=self.bands,
            ring_color=hsl(hue + 0.1, sat, 0.3 + self.pulse * 0.5),
            pt_size=0.07 + self.treble * 0.18 + self.pulse * 0.05,
            pt_color=hsl(hue + 0.55, sat * 0.6, 0.65),
            bloom=self.bloom * (0.6 + self.bass * 0.6 + self.pulse * 0.35),
            hud_alpha=hud_alpha, **self.rot)

    # ------------------------------------------------------------ HUD
    def update_hud(self):
        lines = [
            ("Music Viewer", True),
            (f"Source: {self.audio.device_name}", False),
            (f"Palette: {PALETTES[self.palette][0]}   Sensitivity: {self.sensitivity:.2f}   "
             f"Bloom: {self.bloom:.2f}   Orbit: {'on' if self.auto_orbit else 'off'}   "
             f"{self.fps:.0f} fps", False),
        ]
        if self.show_help or time.perf_counter() < self.help_until:
            lines += [("", False)] + [(h, False) for h in HELP]
        key = (tuple(lines), )
        if key == self._hud_key:
            return
        self._hud_key = key
        surfs = [(self.font_b if bold else self.font).render(txt or " ", True,
                  (232, 236, 255) if bold else (175, 182, 215)) for txt, bold in lines]
        pad = 14
        w = max(s.get_width() for s in surfs) + pad * 2
        h = sum(s.get_height() for s in surfs) + pad * 2
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(panel, (14, 16, 32, 170), panel.get_rect(), border_radius=12)
        pygame.draw.rect(panel, (255, 255, 255, 28), panel.get_rect(), width=1, border_radius=12)
        y = pad
        for s in surfs:
            panel.blit(s, (pad, y))
            y += s.get_height()
        self.scene.set_overlay(pygame.image.tobytes(panel, "RGBA", True), (w, h))

    def poke_hud(self, seconds=3.0):
        self.hud_until = max(self.hud_until, time.perf_counter() + seconds)

    # ------------------------------------------------------------ input
    def handle(self, e):
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.KEYDOWN:
            k = e.key
            if k in (pygame.K_ESCAPE, pygame.K_q):
                return False
            elif k in (pygame.K_f, pygame.K_F11):
                pygame.display.toggle_fullscreen()
            elif k == pygame.K_c:
                self.palette = (self.palette + 1) % len(PALETTES)
                self.hue_drift = 0.0
            elif k == pygame.K_r:
                self.auto_orbit = not self.auto_orbit
            elif k == pygame.K_UP:
                self.sensitivity = min(4.0, self.sensitivity + 0.1)
            elif k == pygame.K_DOWN:
                self.sensitivity = max(0.2, self.sensitivity - 0.1)
            elif k == pygame.K_RIGHTBRACKET:
                self.bloom = min(3.0, self.bloom + 0.1)
            elif k == pygame.K_LEFTBRACKET:
                self.bloom = max(0.0, self.bloom - 0.1)
            elif k == pygame.K_h:
                self.show_help = not self.show_help
            self.poke_hud()
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            self.dragging = True
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.dragging = False
        elif e.type == pygame.MOUSEMOTION and self.dragging:
            self.yaw -= e.rel[0] * 0.005
            self.pitch = max(-1.3, min(1.3, self.pitch + e.rel[1] * 0.005))
        elif e.type == pygame.MOUSEWHEEL:
            self.dist = max(4.0, min(40.0, self.dist * 0.9 ** e.y))
        elif e.type in (pygame.VIDEORESIZE, pygame.WINDOWSIZECHANGED):
            self.scene.resize(pygame.display.get_window_size())
        return True

    # ------------------------------------------------------------ loop
    def run(self, screenshot=None, seconds=None):
        clock = pygame.time.Clock()
        start = last = time.perf_counter()
        running = True
        while running:
            for e in pygame.event.get():
                running = self.handle(e) and running
            now = time.perf_counter()
            dt, last = min(now - last, 0.05), now
            t = now - start
            self.fps = 0.95 * self.fps + 0.05 * (1 / max(dt, 1e-4))

            self.update(dt, t)
            show = self.show_help or now < self.hud_until
            hud_alpha = 1.0 if show else max(0.0, 1.0 - (now - self.hud_until) / 0.6)
            if hud_alpha > 0:
                self.update_hud()
            self.scene.render(self.frame_state(t, hud_alpha))
            pygame.display.flip()
            clock.tick(240)

            if seconds is not None and t > seconds:
                if screenshot:
                    w, h = self.scene.size
                    img = self.ctx.screen.read(viewport=(0, 0, w, h), components=3)
                    surf = pygame.image.frombytes(img, (w, h), "RGB", True)
                    pygame.image.save(surf, screenshot)
                running = False
        self.audio.close()
        pygame.quit()


def main():
    ap = argparse.ArgumentParser(description="3D music visualizer for system audio")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--screenshot", help=argparse.SUPPRESS)
    ap.add_argument("--seconds", type=float, help=argparse.SUPPRESS)
    args = ap.parse_args()
    Visualizer(args).run(args.screenshot, args.seconds)


if __name__ == "__main__":
    main()
