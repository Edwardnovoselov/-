#!/usr/bin/env python3
"""Короткие вертикальные видео для изучения английских слов (TikTok / Reels / Shorts).

Форматы:
  cards — русское слово → отсчёт 3 сек → английское слово с транскрипцией и озвучкой
  quiz  — «Как переводится X?» с тремя вариантами → отсчёт → правильный ответ

Примеры:
  python3 make_video.py --level B1 --format cards --part 1
  python3 make_video.py --level B2 --format quiz --part 2 --handle @my.english
  python3 make_video.py --level C1 --format quiz --stills 1,4,6   # только кадры PNG для проверки дизайна
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
FONTS = ROOT / "assets" / "fonts"
CACHE = ROOT / "cache"
W, H, FPS, SR = 1080, 1920, 30, 48000

LEVELS = {
    "A1": ("Beginner", (52, 211, 153)),
    "A2": ("Elementary", (34, 211, 238)),
    "B1": ("Intermediate", (251, 191, 36)),
    "B2": ("Upper-Intermediate", (251, 146, 60)),
    "C1": ("Advanced", (244, 114, 182)),
    "C2": ("Proficiency", (167, 139, 250)),
}
VOICE_RU = "ru-RU-SvetlanaNeural"
VOICE_EN = "en-GB-SoniaNeural"

# Рекламный баннер: один раз в середине ролика (ролики до 2 минут), по центру экрана,
# на всю ширину, ускорен в 1.3 раза, зелёный хромакей удалён.
BANNER = ROOT / "assets" / "banner.mp4"
BANNER_SPEED = 1.3
BANNER_KEY = "colorkey=0x00FE00:0.30:0.08,despill=type=green"
BANNER_GAIN_DB = 7.0  # исходник тихий (−25 LUFS): подтягиваем к громкости озвучки
BANNER_PAD_IN, BANNER_PAD_OUT = 0.3, 0.35  # тишина до и после баннера

DISPLAY_XB = FONTS / "InterDisplay-ExtraBold.otf"
DISPLAY_B = FONTS / "InterDisplay-Bold.otf"
TEXT_SB = FONTS / "Inter-SemiBold.otf"
IPA = FONTS / "NotoSans-Medium.ttf"

DARK = (17, 19, 24)
GLASS = (12, 16, 24)
GREEN = (34, 197, 94)

FORMATS = {
    # слов в ролике, длительность отсчёта, слов на один фоновый клип
    "cards": {"per_video": 10, "countdown": 3.0, "per_clip": 2},
    "quiz": {"per_video": 8, "countdown": 4.5, "per_clip": 2},
}


# ---------------------------------------------------------------- слова


@dataclass
class Word:
    en: str
    ipa: str
    ru: str
    wrong: list[str]


def load_words(level: str) -> list[Word]:
    words = []
    for line in (ROOT / "words" / f"{level}.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        en, ipa, ru, w1, w2 = (p.strip() for p in line.split("|"))
        words.append(Word(en, ipa, ru, [w1, w2]))
    # фиксированное перемешивание: части не пересекаются и стабильны между запусками
    random.Random(f"words-{level}").shuffle(words)
    return words


def spoken(ru: str) -> str:
    """Текст для озвучки: без пояснений в скобках («высокий (о человеке)» → «высокий»)."""
    return re.sub(r"\s*\([^)]*\)", "", ru).strip()


def pick_words(level: str, fmt: str, part: int) -> list[Word]:
    words = load_words(level)
    n = FORMATS[fmt]["per_video"]
    start = (part - 1) * n
    return [words[(start + i) % len(words)] for i in range(n)]


# ---------------------------------------------------------------- звук


def _edge_tts(text: str, voice: str, rate: str, dst: Path) -> None:
    ca = os.environ.get("SSL_CERT_FILE")
    if ca:
        import certifi

        certifi.where = lambda: ca  # edge_tts берёт CA из certifi при импорте
    import edge_tts

    asyncio.run(edge_tts.Communicate(text, voice, rate=rate).save(str(dst)))


def decode(path: Path, af: str = "anull") -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-af", af, "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def media_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return float(out)


def trim(pcm: np.ndarray, thr: float = 0.012) -> np.ndarray:
    idx = np.flatnonzero(np.abs(pcm) > thr)
    if not len(idx):
        return pcm
    pad = int(0.02 * SR)
    return pcm[max(0, idx[0] - pad) : idx[-1] + pad]


def tts(text: str, voice: str, rate: str = "+0%") -> np.ndarray:
    key = hashlib.sha1(f"{voice}|{rate}|{text}".encode()).hexdigest()[:16]
    mp3 = CACHE / "tts" / f"{key}.mp3"
    if not mp3.exists() or mp3.stat().st_size == 0:
        mp3.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            try:
                _edge_tts(text, voice, rate, mp3)
                if mp3.stat().st_size > 0:
                    break
            except Exception as e:  # noqa: BLE001
                print(f"tts retry {attempt + 1}: {e}", file=sys.stderr)
        else:
            raise RuntimeError(f"TTS не удался: {text!r}")
    return trim(decode(mp3))


def _decay(n: int, tau: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (tau * SR))


def sfx_tick(high: bool) -> np.ndarray:
    """Короткий «деревянный» щелчок часов."""
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    f1, f2 = (2100, 4700) if high else (1700, 3900)
    body = np.sin(2 * np.pi * f1 * t) * _decay(n, 0.009) + 0.5 * np.sin(2 * np.pi * f2 * t) * _decay(n, 0.004)
    click = np.random.default_rng(7).standard_normal(n) * _decay(n, 0.0012)
    return (0.7 * body + 0.35 * np.diff(click, prepend=0)).astype(np.float32)


def sfx_pop() -> np.ndarray:
    """Мягкий «буль» при появлении ответа."""
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    freq = 420 + 520 * (1 - np.exp(-t / 0.03))
    phase = 2 * np.pi * np.cumsum(freq) / SR
    env = (1 - np.exp(-t / 0.003)) * np.exp(-t / 0.05)
    return (np.sin(phase) * env).astype(np.float32)


def sfx_chime() -> np.ndarray:
    """Двухнотный «правильно!» (до–ми)."""
    n = int(0.9 * SR)
    out = np.zeros(n, dtype=np.float32)
    for delay, f in ((0.0, 1046.5), (0.085, 1318.5)):
        k = int(delay * SR)
        m = n - k
        t = np.arange(m) / SR
        env = (1 - np.exp(-t / 0.004)) * np.exp(-t / 0.22)
        tone = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) + 0.08 * np.sin(6 * np.pi * f * t)
        out[k:] += (tone * env).astype(np.float32)
    return out / np.abs(out).max()


class Mixer:
    def __init__(self) -> None:
        self.events: list[tuple[float, np.ndarray, float]] = []

    def add(self, t: float, pcm: np.ndarray, gain_db: float = 0.0) -> None:
        self.events.append((t, pcm, 10 ** (gain_db / 20)))

    def render(self, duration: float) -> np.ndarray:
        out = np.zeros(int(duration * SR) + SR, dtype=np.float32)
        for t, pcm, g in self.events:
            k = int(t * SR)
            seg = pcm[: max(0, len(out) - k)]
            out[k : k + len(seg)] += seg * g
        out = out[: int(duration * SR)]
        peak = np.abs(out).max() or 1.0
        return out * (0.89 / peak)  # пик ≈ −1 dBFS


# ---------------------------------------------------------------- графика


@lru_cache(maxsize=None)
def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def ease_out_cubic(x: float) -> float:
    return 1 - (1 - x) ** 3


def ease_in_cubic(x: float) -> float:
    return x**3


def ease_out_back(x: float, s: float = 1.6) -> float:
    x -= 1
    return 1 + x * x * ((s + 1) * x + s)


def lerp(a: float, b: float, x: float) -> float:
    return a + (b - a) * x


def rr_mask(w: int, h: int, r: int, ss: int = 3) -> Image.Image:
    """Сглаженная маска скруглённого прямоугольника."""
    m = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), r * ss, fill=255)
    return m.resize((w, h), Image.LANCZOS)


def shadow_layer(size: tuple[int, int], box: tuple[int, int, int, int], r: int, alpha: int, blur: int) -> Image.Image:
    sh = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle(box, r, fill=(0, 0, 0, alpha))
    return sh.filter(ImageFilter.GaussianBlur(blur))


def panel(w: int, h: int, r: int, fill: tuple, border: tuple | None = None, shadow: int = 90) -> tuple[Image.Image, int]:
    """Плашка со скруглением, рамкой и мягкой тенью. Возвращает (картинка, отступ под тень)."""
    m = 44
    out = Image.new("RGBA", (w + 2 * m, h + 2 * m), (0, 0, 0, 0))
    if shadow:
        out.alpha_composite(shadow_layer(out.size, (m, m + 12, m + w, m + h + 12), r, shadow, 20))
    mask = rr_mask(w, h, r)
    body = Image.new("RGBA", (w, h), fill[:3] + (255,))
    a = fill[3] if len(fill) == 4 else 255
    body.putalpha(mask.point(lambda v: v * a // 255))
    if border:
        inner = rr_mask(w - 2 * border[0], h - 2 * border[0], max(1, r - border[0]))
        ring = mask.copy()
        ring.paste(0, (border[0], border[0]), inner)
        col = Image.new("RGBA", (w, h), border[1][:3] + (0,))
        col.putalpha(ring.point(lambda v: v * border[1][3] // 255))
        body.alpha_composite(col)
    out.alpha_composite(body, (m, m))
    return out, m


def text_size(f: ImageFont.FreeTypeFont, s: str) -> tuple[int, int]:
    asc, desc = f.getmetrics()
    return math.ceil(f.getlength(s)), asc + desc


def draw_text(img: Image.Image, xy: tuple[float, float], s: str, f: ImageFont.FreeTypeFont, fill: tuple, anchor: str = "mm") -> None:
    ImageDraw.Draw(img).text(xy, s, font=f, fill=fill, anchor=anchor)


def fit_lines(s: str, path: Path, max_w: int, start: int, single_min: int, min_size: int) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """Крупно в одну строку; если не влезает — в две строки по пробелу."""
    for size in range(start, single_min - 1, -2):
        if font(path, size).getlength(s) <= max_w:
            return font(path, size), [s]
    parts = s.split(" ")
    if len(parts) > 1:
        splits = [(" ".join(parts[:i]), " ".join(parts[i:])) for i in range(1, len(parts))]
        for size in range(start, min_size - 1, -2):
            f = font(path, size)
            best = min(splits, key=lambda p: max(f.getlength(p[0]), f.getlength(p[1])))
            if max(f.getlength(best[0]), f.getlength(best[1])) <= max_w:
                return f, list(best)
    return font(path, min_size), [s]


def paste(img: Image.Image, sprite: Image.Image, cx: float, cy: float, scale: float = 1.0, opacity: float = 1.0) -> None:
    if opacity <= 0.004:
        return
    sp = sprite
    if abs(scale - 1) > 0.002:
        sp = sp.resize((max(1, round(sp.width * scale)), max(1, round(sp.height * scale))), Image.BILINEAR)
    if opacity < 0.996:
        sp = sp.copy()
        sp.putalpha(sp.getchannel("A").point(lambda v: int(v * opacity)))
    x, y = round(cx - sp.width / 2), round(cy - sp.height / 2)
    # обрезаем по краям кадра
    sx, sy = max(0, -x), max(0, -y)
    if sx or sy or x + sp.width > W or y + sp.height > H:
        sp = sp.crop((sx, sy, min(sp.width, W - x), min(sp.height, H - y)))
        x, y = x + sx, y + sy
    img.alpha_composite(sp, (x, y))


# ---------------------------------------------------------------- элементы


def make_vignette(handle: str) -> Image.Image:
    """Лёгкое затемнение: сильнее сверху и снизу, чтобы текст читался на любом фоне."""
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    top = np.clip(1 - y / 0.30, 0, 1) ** 1.6 * 0.50
    bottom = np.clip((y - 0.68) / 0.32, 0, 1) ** 1.4 * 0.50
    a = np.clip(0.10 + top + bottom, 0, 1)
    alpha = (np.repeat(a, W, axis=1) * 255).astype(np.uint8)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    img.putalpha(Image.fromarray(alpha, "L"))
    if handle:
        draw_text(img, (W / 2, 1500), handle, font(TEXT_SB, 34), (255, 255, 255, 150))
    return img


def make_badge(level: str) -> Image.Image:
    name, color = LEVELS[level]
    f1, f2 = font(DISPLAY_XB, 50), font(TEXT_SB, 36)
    w1 = text_size(f1, level)[0] + 52
    w2 = text_size(f2, name)[0] + 52
    h = 84
    w = w1 + w2
    out, m = panel(w, h, h // 2, GLASS + (150,), border=(2, (255, 255, 255, 50)), shadow=70)
    left = Image.new("RGBA", (w1, h), color + (255,))
    mask = rr_mask(w, h, h // 2).crop((0, 0, w1, h))
    left.putalpha(mask)
    out.alpha_composite(left, (m, m))
    draw_text(out, (m + w1 / 2, m + h / 2 + 1), level, f1, DARK)
    draw_text(out, (m + w1 + w2 / 2, m + h / 2 + 1), name, f2, (255, 255, 255, 240))
    return out


def make_label(s: str) -> Image.Image:
    f = font(TEXT_SB, 40)
    tw, th = text_size(f, s)
    img = Image.new("RGBA", (tw + 80, th + 60), (0, 0, 0, 0))
    sh = img.copy()
    draw_text(sh, (img.width / 2, img.height / 2 + 3), s, f, (0, 0, 0, 200))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(8)))
    draw_text(img, (img.width / 2, img.height / 2), s, f, (255, 255, 255, 245))
    return img


def make_dots(n: int, current: int, color: tuple) -> Image.Image:
    d, gap, wide = 14, 14, 40
    total = (n - 1) * (d + gap) + wide
    ss = 3
    img = Image.new("RGBA", ((total + 20) * ss, (d + 20) * ss), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    x = 10
    for i in range(n):
        w = wide if i == current else d
        if i == current:
            fill = color + (255,)
        elif i < current:
            fill = (255, 255, 255, 235)
        else:
            fill = (255, 255, 255, 90)
        dr.rounded_rectangle((x * ss, 10 * ss, (x + w) * ss, (10 + d) * ss), d * ss // 2, fill=fill)
        x += w + gap
    return img.resize((img.width // ss, img.height // ss), Image.LANCZOS)


def make_ru_card(ru: str) -> Image.Image:
    s = ru[0].upper() + ru[1:]
    f, lines = fit_lines(s, DISPLAY_B, 780, 104, 80, 64)
    lh = int(f.size * 1.14)
    tw = max(text_size(f, ln)[0] for ln in lines)
    w, h = max(tw + 120, 420), lh * len(lines) + 70
    out, m = panel(w, h, 40, (255, 255, 255, 246), shadow=110)
    y0 = m + h / 2 - lh * (len(lines) - 1) / 2
    for i, ln in enumerate(lines):
        draw_text(out, (m + w / 2, y0 + i * lh + 2), ln, f, DARK)
    return out


def make_en_card(en: str, ipa: str, color: tuple) -> Image.Image:
    f, lines = fit_lines(en, DISPLAY_XB, 780, 100, 72, 64)
    fi = font(IPA, 46)
    ipa_s = f"[{ipa}]"
    tw = max([text_size(f, ln)[0] for ln in lines] + [text_size(fi, ipa_s)[0]])
    lh = int(f.size * 1.12)
    w = max(tw + 120, 420)
    h = lh * len(lines) + 62 + 88
    out, m = panel(w, h, 40, GLASS + (205,), border=(3, color + (230,)), shadow=110)
    y0 = m + 34 + lh / 2
    for i, ln in enumerate(lines):
        draw_text(out, (m + w / 2, y0 + i * lh), ln, f, color + (255,))
    draw_text(out, (m + w / 2, m + h - 34 - 30), ipa_s, fi, (255, 255, 255, 225))
    return out


def make_ipa_chip(en: str, ipa: str) -> Image.Image:
    f1, f2 = font(DISPLAY_B, 48), font(IPA, 44)
    ipa_s = f"[{ipa}]"
    w1, w2 = text_size(f1, en)[0], text_size(f2, ipa_s)[0]
    gap = 22
    w, h = w1 + gap + w2 + 84, 96
    out, m = panel(w, h, h // 2, GLASS + (190,), border=(2, (255, 255, 255, 60)), shadow=80)
    x = m + 42
    draw_text(out, (x, m + h / 2 + 1), en, f1, (255, 255, 255, 255), anchor="lm")
    draw_text(out, (x + w1 + gap, m + h / 2 + 1), ipa_s, f2, (255, 255, 255, 200), anchor="lm")
    return out


def make_question(ru: str, color: tuple) -> Image.Image:
    f1 = font(TEXT_SB, 44)
    s = f"«{ru}»"
    f2, lines = fit_lines(s, DISPLAY_XB, 760, 92, 70, 60)
    lh = int(f2.size * 1.14)
    tw = max([text_size(f2, ln)[0] for ln in lines] + [text_size(f1, "Как переводится")[0]])
    w = max(tw + 120, 640)
    h = 62 + 56 + lh * len(lines) + 46
    out, m = panel(w, h, 40, (255, 255, 255, 246), shadow=110)
    draw_text(out, (m + w / 2, m + 62), "Как переводится", f1, (96, 102, 116, 255))
    y0 = m + 62 + 56 + lh / 2
    for i, ln in enumerate(lines):
        cy = y0 + i * lh
        lw = text_size(f2, ln)[0]
        # маркерное выделение цветом уровня
        hl = Image.new("RGBA", out.size, (0, 0, 0, 0))
        ImageDraw.Draw(hl).rounded_rectangle(
            (m + w / 2 - lw / 2 - 12, cy + f2.size * 0.02, m + w / 2 + lw / 2 + 12, cy + f2.size * 0.42), 10, fill=color + (150,)
        )
        out.alpha_composite(hl)
        draw_text(out, (m + w / 2, cy), ln, f2, DARK)
    return out


OPT_W, OPT_H = 820, 116


def make_option(letter: str, text: str, state: str) -> Image.Image:
    """state: normal | correct"""
    if state == "correct":
        out, m = panel(OPT_W, OPT_H, OPT_H // 2, GREEN + (245,), border=(3, (255, 255, 255, 120)), shadow=110)
    else:
        out, m = panel(OPT_W, OPT_H, OPT_H // 2, GLASS + (170,), border=(2, (255, 255, 255, 70)), shadow=80)
    d = 76
    cx, cy = m + 20 + d / 2, m + OPT_H / 2
    ss = 3
    circ = Image.new("RGBA", (d * ss, d * ss), (0, 0, 0, 0))
    cd = ImageDraw.Draw(circ)
    if state == "correct":
        cd.ellipse((0, 0, d * ss - 1, d * ss - 1), fill=(255, 255, 255, 255))
        pts = [(0.27, 0.52), (0.43, 0.68), (0.74, 0.35)]
        cd.line([(x * d * ss, y * d * ss) for x, y in pts], fill=GREEN + (255,), width=8 * ss, joint="curve")
        for x, y in (pts[0], pts[2]):
            r = 4 * ss
            cd.ellipse((x * d * ss - r, y * d * ss - r, x * d * ss + r, y * d * ss + r), fill=GREEN + (255,))
    else:
        cd.ellipse((0, 0, d * ss - 1, d * ss - 1), fill=(255, 255, 255, 46))
    circ = circ.resize((d, d), Image.LANCZOS)
    out.alpha_composite(circ, (round(cx - d / 2), round(cy - d / 2)))
    if state != "correct":
        draw_text(out, (cx, cy + 1), letter, font(DISPLAY_B, 38), (255, 255, 255, 255))
    label = text[0].upper() + text[1:]
    f = font(TEXT_SB, 52)
    while f.getlength(label) > OPT_W - 160 and f.size > 34:
        f = font(TEXT_SB, f.size - 2)
    draw_text(out, (m + 20 + d + 30, cy + 1), label, f, (255, 255, 255, 255), anchor="lm")
    return out


RING_R, RING_STROKE, RING_SS = 92, 13, 3
RING_SIZE = 2 * RING_R + 2 * RING_STROKE + 60


def _ring_box(r: float) -> tuple:
    c = RING_SIZE * RING_SS / 2
    return (c - r, c - r, c + r, c + r)


@lru_cache(maxsize=1)
def ring_base() -> Image.Image:
    """Неизменная часть таймера: тень, подложка и серая дорожка."""
    R, stroke, ss = RING_R, RING_STROKE, RING_SS
    img = Image.new("RGBA", (RING_SIZE * ss, RING_SIZE * ss), (0, 0, 0, 0))
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse(_ring_box((R + stroke) * ss), fill=(0, 0, 0, 90))
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(14 * ss)))
    d = ImageDraw.Draw(img)
    d.ellipse(_ring_box((R - stroke / 2) * ss), fill=GLASS + (150,))
    d.ellipse(_ring_box((R + stroke / 2) * ss), outline=(255, 255, 255, 70), width=stroke * ss)
    return img.resize((RING_SIZE, RING_SIZE), Image.LANCZOS)


def make_ring(remaining: float, number: int, color: tuple, num_scale: float) -> Image.Image:
    R, stroke, ss = RING_R, RING_STROKE, RING_SS
    img = ring_base().copy()
    if remaining > 0.001:
        arc = Image.new("RGBA", (RING_SIZE * ss, RING_SIZE * ss), (0, 0, 0, 0))
        d = ImageDraw.Draw(arc)
        c = RING_SIZE * ss / 2
        end = -90 + 360 * remaining
        d.arc(_ring_box((R + stroke / 2) * ss), -90, end, fill=color + (255,), width=stroke * ss)
        for ang in (-90, end):  # скруглённые концы дуги
            a = math.radians(ang)
            px, py = c + R * ss * math.cos(a), c + R * ss * math.sin(a)
            rr = stroke * ss / 2
            d.ellipse((px - rr, py - rr, px + rr, py + rr), fill=color + (255,))
        img.alpha_composite(arc.resize((RING_SIZE, RING_SIZE), Image.BOX))
    draw_text(img, (RING_SIZE / 2, RING_SIZE / 2 + 2), str(number), font(DISPLAY_XB, max(10, round(88 * num_scale))), (255, 255, 255, 255))
    return img


def make_end_card(title: str, sub: str, color: tuple) -> tuple[Image.Image, Image.Image]:
    f1, f2 = font(DISPLAY_XB, 70), font(TEXT_SB, 44)
    _, lines = fit_lines(title, DISPLAY_XB, 820, 70, 70, 60)
    lh = 84
    tw = max([text_size(f1, ln)[0] for ln in lines] + [text_size(f2, sub)[0]])
    w = tw + 130
    h = 70 + lh * len(lines) + 70 + 40
    card, m = panel(w, h, 44, (255, 255, 255, 246), shadow=110)
    for i, ln in enumerate(lines):
        draw_text(card, (m + w / 2, m + 70 + lh / 2 + i * lh - 18), ln, f1, DARK)
    draw_text(card, (m + w / 2, m + h - 62), sub, f2, (96, 102, 116, 255))
    fp = font(DISPLAY_B, 40)
    s = "Подпишись — новые слова каждый день"
    pw = text_size(fp, s)[0] + 80
    pill, pm = panel(pw, 92, 46, color + (255,), shadow=90)
    draw_text(pill, (pm + pw / 2, pm + 47), s, fp, DARK)
    return card, pill


# ---------------------------------------------------------------- сцена


@dataclass
class Item:
    word: Word
    t0: float = 0.0
    cd0: float = 0.0
    reveal: float = 0.0
    end: float = 0.0
    sprites: dict = field(default_factory=dict)
    options: list = field(default_factory=list)
    correct: int = 0


class Video:
    def __init__(
        self, level: str, fmt: str, part: int, handle: str = "", preset: str = "superfast", banner: Path | None = None
    ) -> None:
        self.level, self.fmt, self.part, self.preset, self.banner = level, fmt, part, preset, banner
        self.name, self.color = LEVELS[level]
        self.cfg = FORMATS[fmt]
        self.items = [Item(w) for w in pick_words(level, fmt, part)]
        self.mix = Mixer()
        self.vignette = make_vignette(handle)
        self.badge = make_badge(level)
        self.label = make_label("Переведи на английский" if fmt == "cards" else "Выбери правильный перевод")
        self.headers: dict[int, Image.Image] = {}
        self.banner_after = None
        self.build()
        if banner:
            # баннер ставим на границу слов, ближайшую к середине итогового ролика,
            # так что середина ролика всегда попадает внутрь баннера
            self.banner_len = media_duration(banner) / BANNER_SPEED
            plain = self.duration

            def placement(k: int) -> tuple[float, float]:
                """Самая короткая пауза перед баннером, при которой середина ролика попадает в баннер
                с запасом 0.5 с от краёв; второе значение — промах (0, если условие выполнено)."""
                end, L, margin = self.items[k].end, self.banner_len, 0.5
                lo = max(BANNER_PAD_IN, plain - L + BANNER_PAD_OUT - 2 * end + 2 * margin)
                hi = min(1.6, plain + L + BANNER_PAD_OUT - 2 * end - 2 * margin)
                pad = lo if lo <= hi else clamp(plain + BANNER_PAD_OUT - 2 * end, BANNER_PAD_IN, 1.6)
                center = end + pad + L / 2
                middle = (plain + pad + L + BANNER_PAD_OUT) / 2
                miss = 0.0 if lo <= hi else abs(center - middle)
                return pad, miss

            self.banner_after = min(range(1, len(self.items) - 1), key=lambda k: (placement(k)[1], placement(k)[0]))
            self.banner_pad = placement(self.banner_after)[0]
            self.mix = Mixer()
            self.build()

    # --- раскладка по времени и звук
    def build(self) -> None:
        tick_hi, tick_lo = sfx_tick(True), sfx_tick(False)
        pop, chime = sfx_pop(), sfx_chime()
        cd_len = self.cfg["countdown"]
        self.rng = random.Random(f"{self.level}-{self.fmt}-{self.part}")
        t = 0.0
        for i, it in enumerate(self.items):
            w = it.word
            it.t0 = t
            if self.fmt == "cards":
                ru = tts(spoken(w.ru), VOICE_RU)
                self.mix.add(t + 0.12, ru)
                it.cd0 = max(t + 0.75, t + 0.12 + len(ru) / SR + 0.15)
                it.sprites["ru"] = make_ru_card(w.ru)
                it.sprites["en"] = make_en_card(w.en, w.ipa, self.color)
            else:
                q = tts(f"Как переводится «{spoken(w.ru)}»?", VOICE_RU)
                self.mix.add(t + 0.15, q)
                it.cd0 = t + 0.9
                opts = [w.en] + w.wrong
                self.rng.shuffle(opts)
                it.options, it.correct = opts, opts.index(w.en)
                it.sprites["q"] = make_question(w.ru, self.color)
                it.sprites["opt"] = [make_option("ABC"[k], o, "normal") for k, o in enumerate(opts)]
                it.sprites["ok"] = make_option("ABC"[it.correct], w.en, "correct")
                it.sprites["chip"] = make_ipa_chip(w.en, w.ipa)
            it.reveal = it.cd0 + cd_len
            for k in range(int(round(cd_len * 2))):
                self.mix.add(it.cd0 + k * 0.5, tick_hi if k % 2 == 0 else tick_lo, -17)
            en = tts(w.en, VOICE_EN, rate="-8%")
            if self.fmt == "cards":
                self.mix.add(it.reveal, pop, -12)
                self.mix.add(it.reveal + 0.12, en)
                it.end = max(it.reveal + 1.55, it.reveal + 0.12 + len(en) / SR + 0.5)
            else:
                self.mix.add(it.reveal, chime, -11)
                self.mix.add(it.reveal + 0.3, en)
                it.end = max(it.reveal + 1.75, it.reveal + 0.3 + len(en) / SR + 0.5)
            t = it.end
            if i == self.banner_after:
                self.banner_t0 = t + self.banner_pad
                self.mix.add(self.banner_t0, decode(self.banner, f"atempo={BANNER_SPEED}"), BANNER_GAIN_DB)
                t = self.banner_t0 + self.banner_len + BANNER_PAD_OUT
        # финальная карточка
        self.end_t0 = t
        n = len(self.items)
        title = "Сколько слов ты знал?" if self.fmt == "cards" else f"Сколько из {n} верно?"
        self.end_card, self.end_pill = make_end_card(title, "Напиши в комментариях", self.color)
        outro = tts(f"{title} Напиши в комментариях!", VOICE_RU)
        self.mix.add(t + 0.15, outro)
        self.duration = t + max(2.8, 0.15 + len(outro) / SR + 0.6)

    # --- фон
    def header(self, idx: int) -> Image.Image:
        if idx not in self.headers:
            img = self.vignette.copy()
            paste(img, self.badge, W / 2, 285)
            paste(img, self.label, W / 2, 382)
            paste(img, make_dots(len(self.items), idx, self.color), W / 2, 446)
            self.headers[idx] = img
        return self.headers[idx]

    def current(self, t: float) -> int:
        for i, it in enumerate(self.items):
            if t < it.end:
                return i
        return len(self.items)

    # --- кадр
    def static_key(self, t: float) -> tuple | None:
        """Ключ для кадров без анимации: одинаковый ключ — одинаковая картинка, её не перерисовываем."""
        i = self.current(t)
        if i == len(self.items):
            return ("end",) if t >= self.end_t0 + 0.85 else None
        it = self.items[i]
        if t < it.t0:  # пауза под баннер: на слое интерфейса только шапка
            return ("gap", i)
        appeared = it.t0 + (0 if i == 0 else 0.7)
        if appeared <= t < it.cd0 - 0.25:
            return ("ask", i)
        if it.reveal + 0.55 <= t < it.end - 0.3:
            return ("answer", i)
        return None

    def frame(self, t: float) -> Image.Image:
        i = self.current(t)
        img = self.header(i).copy()
        if i == len(self.items):
            self.draw_end(img, t)
        elif self.fmt == "cards":
            self.draw_cards(img, t, self.items[i], first=i == 0)
        else:
            self.draw_quiz(img, t, self.items[i], first=i == 0)
        return img

    def ring(self, img: Image.Image, t: float, it: Item, cy: float) -> None:
        cd_len = self.cfg["countdown"]
        if not (it.cd0 - 0.25 <= t < it.reveal + 0.3):
            return
        a_in = ease_out_cubic(clamp((t - (it.cd0 - 0.25)) / 0.25))
        a_out = ease_in_cubic(clamp((t - it.reveal) / 0.25))
        remaining = clamp(1 - (t - it.cd0) / cd_len)
        left = clamp(it.reveal - t, 0, cd_len)
        num = max(1, math.ceil(left - 1e-6))
        since_change = num - left if t >= it.cd0 else 1.0  # цифра «подпрыгивает» при смене
        pop = 1 + 0.28 * (1 - ease_out_cubic(clamp(since_change / 0.28)))
        sprite = make_ring(remaining, num, self.color, pop)
        paste(img, sprite, W / 2, cy, scale=lerp(0.85, 1, a_in) * lerp(1, 0.8, a_out), opacity=a_in * (1 - a_out))

    def draw_cards(self, img: Image.Image, t: float, it: Item, first: bool) -> None:
        a_in = 1.0 if first else clamp((t - it.t0) / 0.42)
        a_out = clamp((t - (it.end - 0.3)) / 0.3)
        out_s, out_o = lerp(1, 0.94, ease_in_cubic(a_out)), 1 - ease_in_cubic(a_out)
        paste(
            img,
            it.sprites["ru"],
            W / 2,
            760,
            scale=lerp(0.82, 1, ease_out_back(a_in)) * out_s,
            opacity=ease_out_cubic(a_in) * out_o,
        )
        self.ring(img, t, it, 1080)
        if t >= it.reveal:
            a = clamp((t - it.reveal) / 0.45)
            paste(
                img,
                it.sprites["en"],
                W / 2,
                1080 + lerp(46, 0, ease_out_cubic(a)),
                scale=lerp(0.9, 1, ease_out_back(a)) * out_s,
                opacity=ease_out_cubic(a) * out_o,
            )

    def draw_quiz(self, img: Image.Image, t: float, it: Item, first: bool) -> None:
        a_out = clamp((t - (it.end - 0.3)) / 0.3)
        out_s, out_o = lerp(1, 0.96, ease_in_cubic(a_out)), 1 - ease_in_cubic(a_out)
        a_q = 1.0 if first else clamp((t - it.t0) / 0.42)
        paste(img, it.sprites["q"], W / 2, 690, scale=lerp(0.85, 1, ease_out_back(a_q)) * out_s, opacity=ease_out_cubic(a_q) * out_o)
        revealed = t >= it.reveal
        a_rev = clamp((t - it.reveal) / 0.4)
        for k in range(3):
            y = 930 + k * 138
            a = 1.0 if first else clamp((t - it.t0 - 0.12 - 0.08 * k) / 0.38)
            dy = lerp(36, 0, ease_out_cubic(a))
            o = ease_out_cubic(a) * out_o
            if revealed and k == it.correct:
                bump = 1 + 0.06 * math.sin(math.pi * clamp((t - it.reveal) / 0.4))
                paste(img, it.sprites["opt"][k], W / 2, y, scale=out_s, opacity=o * (1 - a_rev))
                paste(img, it.sprites["ok"], W / 2, y, scale=bump * out_s, opacity=o * ease_out_cubic(a_rev))
            else:
                dim = lerp(1, 0.38, ease_out_cubic(a_rev)) if revealed else 1
                paste(img, it.sprites["opt"][k], W / 2, y + dy, scale=out_s, opacity=o * dim)
        self.ring(img, t, it, 1380)
        if revealed:
            a = clamp((t - it.reveal - 0.1) / 0.4)
            paste(
                img,
                it.sprites["chip"],
                W / 2,
                1380 + lerp(30, 0, ease_out_cubic(a)),
                scale=lerp(0.92, 1, ease_out_back(a)) * out_s,
                opacity=ease_out_cubic(a) * out_o,
            )

    def draw_end(self, img: Image.Image, t: float) -> None:
        a = clamp((t - self.end_t0) / 0.45)
        paste(img, self.end_card, W / 2, 820, scale=lerp(0.85, 1, ease_out_back(a)), opacity=ease_out_cubic(a))
        b = clamp((t - self.end_t0 - 0.35) / 0.45)
        paste(img, self.end_pill, W / 2, 1080 + lerp(40, 0, ease_out_cubic(b)), scale=lerp(0.9, 1, ease_out_back(b)), opacity=ease_out_cubic(b))

    # --- фоновые клипы
    def plan_background(self) -> list[dict]:
        clips = json.loads((ROOT / "bg" / "clips.json").read_text())
        allowed = []
        for ln in (ROOT / "bg" / "curated.txt").read_text().splitlines():
            vid = ln.split("#")[0].strip()
            if vid and vid in clips and (ROOT / "bg" / "clips" / f"{vid}.mp4").exists():
                allowed.append(vid)
        per = self.cfg["per_clip"]
        bounds = [self.items[k].t0 for k in range(per, len(self.items), per)]
        edges = [0.0] + bounds + [self.duration]
        fade = 0.6
        pool = allowed[:]
        self.rng.shuffle(pool)
        segs = []
        for k in range(len(edges) - 1):
            start = edges[k] - (fade / 2 if k else 0)
            end = edges[k + 1] + (fade / 2 if k < len(edges) - 2 else 0)
            need = end - start
            pick = next((v for v in pool if clips[v]["dur"] >= need + 0.4), None)
            if pick is None:  # все длинные уже использованы — берём самый длинный и зацикливаем
                pick = max(allowed, key=lambda v: clips[v]["dur"])
            else:
                pool.remove(pick)
            c = clips[pick]
            off = self.rng.uniform(0.2, max(0.2, c["dur"] - need - 0.2))
            segs.append({"id": pick, "w": c["w"], "h": c["h"], "start": start, "len": need, "off": off if c["dur"] >= need + 0.4 else 0, "loop": c["dur"] < need + 0.4})
        self.fade = fade
        return segs

    # --- сборка
    def render(self, out: Path, stills: list[float] | None = None) -> None:
        segs = self.plan_background()
        if stills:
            return self.render_stills(out, segs, stills)
        out.parent.mkdir(parents=True, exist_ok=True)
        wav = out.with_suffix(".wav")
        audio = self.mix.render(self.duration)
        pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        import wave

        with wave.open(str(wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SR)
            wf.writeframes(pcm16.tobytes())

        cmd = ["ffmpeg", "-v", "error", "-y"]
        for s in segs:
            if s["loop"]:
                cmd += ["-stream_loop", "-1"]
            cmd += ["-ss", f"{s['off']:.3f}", "-t", f"{s['len'] + 0.2:.3f}", "-i", str(ROOT / "bg" / "clips" / f"{s['id']}.mp4")]
        ov = len(segs)
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0", "-i", str(wav)]
        if self.banner:
            cmd += ["-i", str(self.banner)]
        parts = []
        for k, s in enumerate(segs):
            crop = "crop=ih*9/16:ih" if s["w"] / s["h"] > 9 / 16 else "crop=iw:iw*16/9"
            parts.append(
                f"[{k}:v]{crop},scale={W}:{H}:flags=lanczos,fps={FPS},setsar=1,format=yuv420p,"
                f"trim=duration={s['len']:.3f},setpts=PTS-STARTPTS,settb=AVTB[b{k}]"
            )
        last = "b0"
        for k in range(1, len(segs)):
            off = segs[k]["start"]
            parts.append(f"[{last}][b{k}]xfade=transition=fade:duration={self.fade}:offset={off:.3f}[x{k}]")
            last = f"x{k}"
        parts.append(f"[{last}]eq=saturation=1.07:contrast=1.03[bg]")
        if self.banner:
            parts.append(
                f"[{ov + 2}:v]setpts=(PTS-STARTPTS)/{BANNER_SPEED}+{self.banner_t0:.3f}/TB,fps={FPS},format=rgba,"
                f"{BANNER_KEY},scale={W}:-2:flags=lanczos,format=yuva420p[bn]"
            )
            parts.append(f"[bg][{ov}:v]overlay=format=auto:shortest=1[ui]")
            parts.append("[ui][bn]overlay=x=(W-w)/2:y=(H-h)/2:eof_action=pass:format=auto,format=yuv420p[v]")
        else:
            parts.append(f"[bg][{ov}:v]overlay=format=auto:shortest=1,format=yuv420p[v]")
        # громкость под соцсети (≈ −15 LUFS), пики срезает лимитер
        parts.append(f"[{ov + 1}:a]volume=6dB,alimiter=limit=0.89:attack=4:release=60:level=false[a]")
        cmd += [
            "-filter_complex", ";".join(parts),
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", self.preset, "-crf", "22", "-maxrate", "5M", "-bufsize", "10M", "-pix_fmt", "yuv420p", "-r", str(FPS),
            "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-ac", "2",
            "-t", f"{self.duration:.3f}", "-movflags", "+faststart", str(out),
        ]  # fmt: skip
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        nframes = int(math.ceil(self.duration * FPS))
        last_key, last = None, b""
        try:
            for k in range(nframes):
                t = k / FPS
                key = self.static_key(t)
                if key is None or key != last_key:
                    last = self.frame(t).tobytes()
                last_key = key
                proc.stdin.write(last)
        except BrokenPipeError:
            pass
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg завершился с ошибкой")
        wav.unlink()
        out.with_suffix(".txt").write_text(self.caption())
        ad = ""
        if self.banner:
            inside = self.banner_t0 <= self.duration / 2 <= self.banner_t0 + self.banner_len
            ad = f", баннер {self.banner_t0:.1f}–{self.banner_t0 + self.banner_len:.1f} c" + ("" if inside else " НЕ В СЕРЕДИНЕ")
        print(f"готово: {out} ({self.duration:.1f} c{ad}, фоны: {', '.join(s['id'] for s in segs)})")

    def render_stills(self, out: Path, segs: list[dict], times: list[float]) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        for t in times:
            s = next((s for s in segs if s["start"] <= t < s["start"] + s["len"]), segs[-1])
            crop = "crop=ih*9/16:ih" if s["w"] / s["h"] > 9 / 16 else "crop=iw:iw*16/9"
            raw = subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{s['off'] + t - s['start']:.2f}", "-i", str(ROOT / "bg" / "clips" / f"{s['id']}.mp4"),
                 "-frames:v", "1", "-vf", f"{crop},scale={W}:{H}:flags=lanczos,eq=saturation=1.07:contrast=1.03", "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"],
                capture_output=True, check=True,
            ).stdout  # fmt: skip
            bg = Image.frombytes("RGBA", (W, H), raw)
            bg.alpha_composite(self.frame(t))
            p = out.with_name(f"{out.stem}_t{t:05.2f}.png")
            bg.convert("RGB").save(p)
            print(p)

    def caption(self) -> str:
        what = "Переведи 10 слов" if self.fmt == "cards" else f"Квиз: {len(self.items)} слов"
        tags = f"#английский #английскийязык #учиманглийский #английскийонлайн #english #{self.level.lower()} #словаанглийский"
        words = ", ".join(f"{it.word.en} — {it.word.ru}" for it in self.items)
        return f"{what} уровня {self.level} ({self.name}). Сколько знаешь? Пиши в комментариях!\n\n{tags}\n\nСлова: {words}\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", required=True, choices=list(LEVELS))
    ap.add_argument("--format", required=True, choices=list(FORMATS))
    ap.add_argument("--part", type=int, default=1, help="номер ролика в серии (сдвигает набор слов)")
    ap.add_argument("--handle", default="", help="подпись канала, например @my.english")
    ap.add_argument("--banner", default=str(BANNER), help="видео-баннер с зелёным фоном; none — без баннера")
    ap.add_argument("--preset", default="superfast", help="пресет x264: superfast — быстро, medium — чуть меньше файл")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--stills", help="вместо видео сохранить кадры на этих секундах, через запятую")
    args = ap.parse_args()
    out = args.out or ROOT / "out" / f"{args.level}_{args.format}_{args.part:02d}.mp4"
    stills = [float(x) for x in args.stills.split(",")] if args.stills else None
    banner = None if args.banner == "none" else Path(args.banner)
    Video(args.level, args.format, args.part, args.handle, args.preset, banner).render(out, stills)


if __name__ == "__main__":
    main()
