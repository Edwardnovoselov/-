"""Собирает каталог бесплатных вертикальных видео с Mixkit и скачивает пейзажи.

Mixkit License: бесплатно, в том числе для коммерческого использования, без атрибуции.

    python3 fetch_backgrounds.py catalog   # обойти страницы и сохранить bg/catalog.json
    python3 fetch_backgrounds.py download  # скачать в 1080p ролики из bg/curated.txt
"""
import html
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

CA = os.environ.get("SSL_CERT_FILE")  # свой CA нужен только за прокси; иначе системные сертификаты
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
ROOT = Path(__file__).parent
BG = ROOT / "bg"
CATALOG = BG / "catalog.json"

# Что нам подходит: пейзажи, природа, города с высоты.
GOOD = re.compile(
    r"\b(landscape|mountain|mountains|lake|ocean|sea|waves?|beach|coast|shore|forest|trees?|woods|"
    r"waterfall|river|stream|sunset|sunrise|clouds?|sky|aerial|drone|valley|hills?|island|"
    r"canyon|desert|dunes?|snow|glacier|fog|mist|meadow|field|flowers?|city|skyline|town|"
    r"village|bridge|lights|night|stars|autumn|leaves|cliff|rocks?|bay|lagoon|palm|nature|"
    r"jungle|park|street|buildings?)\b",
    re.I,
)
# Что не подходит: люди в кадре, еда, предметы крупным планом.
BAD = re.compile(
    r"\b(woman|women|man|men|girl|boy|person|people|couple|hands?|face|baby|child|kids?|dancer|"
    r"dancing|model|portrait|selfie|food|drink|cocktail|coffee|beer|wine|cake|cooking|kitchen|"
    r"phone|laptop|office|fitness|gym|yoga|workout|makeup|hair|fashion|smoke|abstract|neon|"
    r"tattoo|dog|cat|car interior|surfer|skier|hiker|tourist|traveler|friends|family|bride|"
    r"groom|wedding|player|athlete|runner|cyclist|skateboard|body|legs?|feet|eyes?|lips)\b",
    re.I,
)


def fetch(url: str) -> bytes:
    import ssl

    ctx = ssl.create_default_context(cafile=CA)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
        return r.read()


def parse_page(text: str) -> list[dict]:
    items = []
    for block in re.split(r'<div class="item-grid-card[" ]', text)[1:]:
        m_id = re.search(r'data-item-grid--video-player-item-id-value="(\d+)"', block)
        m_src = re.search(r'<video src="([^"]+)"', block)
        m_title = re.search(r'item-grid-card__title">\s*<a href="([^"]+)">([^<]+)</a>', block) or re.search(
            r'<a href="([^"]+)" class="item-grid-video-player__overlay-link">\s*([^<]+?)\s*</a>', block
        )
        m_desc = re.search(r'item-grid-card__description">([^<]*)</p>', block) or re.search(
            r'class="item-grid-video-player__thumb" alt="([^"]*)"', block
        )
        if not (m_id and m_src and m_title):
            continue
        items.append(
            {
                "id": m_id.group(1),
                "preview": m_src.group(1),
                "page": "https://mixkit.co" + m_title.group(1),
                "title": html.unescape(m_title.group(2).strip()),
                "desc": html.unescape(m_desc.group(1).strip()) if m_desc else "",
                "vertical": "item-grid-video-player--vertical" in block,
            }
        )
    return items


# Горизонтальные категории: кропаем центр в 9:16.
CATEGORIES = ["landscape", "mountain", "aerial", "lake", "waterfall", "forest", "ocean", "sunset", "city"]


def crawl(url_base: str, pages: int, items: list, seen: set) -> None:
    for p in range(1, pages + 1):
        url = url_base + (f"?page={p}" if p > 1 else "")
        try:
            page_items = parse_page(fetch(url).decode("utf-8", "replace"))
        except Exception as e:  # noqa: BLE001
            print(f"{url}: {e}")
            continue
        for it in page_items:
            if it["id"] not in seen:
                seen.add(it["id"])
                items.append(it)
        print(f"{url}: {len(page_items)} items, total {len(items)}")
        time.sleep(0.5)


def catalog(vertical_pages: int = 31, category_pages: int = 4) -> None:
    BG.mkdir(exist_ok=True)
    seen, items = set(), []
    crawl("https://mixkit.co/free-stock-video/discover/vertical/", vertical_pages, items, seen)
    for cat in CATEGORIES:
        crawl(f"https://mixkit.co/free-stock-video/{cat}/", category_pages, items, seen)
    for it in items:
        text = f'{it["title"]} {it["desc"]}'
        it["landscape"] = bool(GOOD.search(text)) and not BAD.search(text)
    CATALOG.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {len(items)} items, landscapes: {sum(i['landscape'] for i in items)}")


def hd_url(item: dict) -> str:
    """Прямая ссылка на 1080p через официальный download-эндпоинт Mixkit."""
    page = fetch(f'https://mixkit.co/free-stock-video/download/{item["id"]}/?context=item&type=1080p').decode()
    m = re.search(r'data-download--modal-url-value="([^"]+)"', page)
    return m.group(1) if m else item["preview"].replace("-360.mp4", "-1080.mp4")


def download(ids: list[str]) -> None:
    by_id = {i["id"]: i for i in json.loads(CATALOG.read_text(encoding="utf-8"))}
    out = BG / "clips"
    out.mkdir(parents=True, exist_ok=True)
    for vid in ids:
        dst = out / f"{vid}.mp4"
        if dst.exists():
            continue
        it = by_id[vid]
        try:
            dst.write_bytes(fetch(hd_url(it)))
            print(f'ok {vid}: {it["title"]}')
        except Exception as e:  # noqa: BLE001
            print(f"fail {vid}: {e}")
        time.sleep(0.3)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "catalog"
    if cmd == "catalog":
        catalog()
    elif cmd == "download":
        # ids из bg/curated.txt (по одному в строке, после # — комментарий)
        lines = (BG / "curated.txt").read_text(encoding="utf-8").splitlines()
        download([ln.split("#")[0].strip() for ln in lines if ln.split("#")[0].strip()])
