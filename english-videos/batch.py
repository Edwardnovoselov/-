#!/usr/bin/env python3
"""Пакетная генерация: все уровни × форматы, без повторов слов внутри формата.

    python3 batch.py                          # всё, что можно собрать из words/*.txt
    python3 batch.py --levels A1,B2 --formats quiz --jobs 2
    python3 batch.py --handle @my.english
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from make_video import FORMATS, LEVELS, ROOT, VOICE_EN, VOICE_RU, load_words, spoken, tts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--formats", default=",".join(FORMATS))
    ap.add_argument("--jobs", type=int, default=2, help="сколько роликов рендерить параллельно")
    ap.add_argument("--handle", default="")
    args = ap.parse_args()

    jobs = []
    for level in args.levels.split(","):
        n_words = len(load_words(level))
        for fmt in args.formats.split(","):
            for part in range(1, n_words // FORMATS[fmt]["per_video"] + 1):
                out = ROOT / "out" / f"{level}_{fmt}_{part:02d}.mp4"
                if not out.exists():
                    jobs.append([sys.executable, str(ROOT / "make_video.py"), "--level", level, "--format", fmt, "--part", str(part), "--handle", args.handle])
    print(f"роликов к рендеру: {len(jobs)}")

    # озвучка заранее и параллельно: потом рендер берёт её из кэша
    phrases = []
    for level in args.levels.split(","):
        for w in load_words(level):
            phrases += [(spoken(w.ru), VOICE_RU, "+0%"), (f"Как переводится «{spoken(w.ru)}»?", VOICE_RU, "+0%"), (w.en, VOICE_EN, "-8%")]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda p: tts(*p), phrases))
    print(f"озвучка готова: {len(phrases)} фраз")

    def run(cmd: list[str]) -> int:
        return subprocess.run(cmd).returncode

    with ThreadPoolExecutor(args.jobs) as ex:
        failed = sum(1 for rc in ex.map(run, jobs) if rc)
    print(f"готово, ошибок: {failed}")


if __name__ == "__main__":
    main()
