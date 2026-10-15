"""Cào danh sách địa điểm F&B từ giao diện web Google Maps (Playwright, headless).

Lưu ý: việc cào giao diện web trái Điều khoản dịch vụ của Google Maps. Module này
cào chậm, chỉ lấy thông tin doanh nghiệp công khai trên trang danh sách (không lấy
review / thông tin người dùng) và cache mọi truy vấn để không gọi lặp. Dữ liệu thô
chỉ dùng cho mục đích học tập, không công khai.

Chiến lược:
- Mỗi truy vấn "<từ khóa>" tại tâm 1 ô lưới, zoom cố định -> cuộn danh sách tới hết
  (Google trả tối đa ~120 kết quả / truy vấn).
- Ô lưới chỉ sinh ở vùng có cửa hàng (dựa trên stores_clean.csv) + các ô kề.
- Nếu 1 truy vấn chạm trần (>= CAP_HIT kết quả) -> thêm 4 tâm lệch 1/4 ô, vẫn zoom 17.

Chạy:
    .venv\\Scripts\\python -m src.scrape_gmaps --max-queries 20          # thử
    .venv\\Scripts\\python -m src.scrape_gmaps                           # chạy đủ (resume được)
    .venv\\Scripts\\python -m src.scrape_gmaps --merge-only              # gộp cache -> CSV
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from src import geo

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "gmaps"
CACHE_DIR = OUT_DIR / "cache"

KEYWORDS = ["quán ăn", "nhà hàng", "quán cà phê", "trà sữa", "tiệm bánh",
            "ăn vặt", "quán nhậu", "cửa hàng thực phẩm"]
BASE_ZOOM = 17
BASE_STEP_M = 500
CAP_HIT = 110        # >= số này coi như chạm trần 120 kết quả -> chia nhỏ
MAX_DEPTH = 1        # chia 1 lần: 4 tâm lệch 250m, GIỮ zoom 17
# Thực nghiệm: ở zoom 17, 90% kết quả nằm trong ~800m quanh tâm; ở zoom 18-19 Google lại
# trả cả điểm cách hàng trăm km -> không tăng zoom khi chia ô.

_EXTRACT_JS = r"""
() => Array.from(document.querySelectorAll('a.hfpxzc')).map(a => {
  const card = a.parentElement;
  const star = card.querySelector('span[role="img"]');
  const rows = Array.from(card.querySelectorAll('div.W4Efsd > div.W4Efsd')).map(d => d.innerText);
  return {name: a.getAttribute('aria-label'), href: a.getAttribute('href'),
          star: star ? star.getAttribute('aria-label') : null,
          rows: rows, text: card.innerText};
})
"""


# ---------------------------------------------------------------- parse
def parse_card(c: dict) -> dict:
    href = c.get("href") or ""
    pid = re.search(r"!1s(0x[0-9a-f]+:0x[0-9a-f]+)", href)
    lat = re.search(r"!3d(-?[\d.]+)", href)
    lon = re.search(r"!4d(-?[\d.]+)", href)
    gid = re.search(r"!19s([A-Za-z0-9_-]+)", href)
    rating = None
    if c.get("star"):
        m = re.search(r"([\d]+[,.][\d])", c["star"])
        rating = float(m.group(1).replace(",", ".")) if m else None
    n_reviews = None
    m = re.search(r"\d[,.]\d\s*\(([\d.,]+)\)", c.get("text", ""))
    if m:
        n_reviews = int(re.sub(r"[.,]", "", m.group(1)))
    category, address = None, None
    rows = c.get("rows") or []
    if rows:
        parts = [p.strip() for p in rows[0].split("·")]
        category = parts[0] or None
        address = parts[-1] if len(parts) > 1 else None
    price = None
    m = re.search(r"(₫{1,4}|[\d.,]+\s*[–-]\s*[\d.,]+\s*(?:N|nghìn|k)?\s*₫)", c.get("text", ""))
    if m:
        price = m.group(1)
    status = rows[1] if len(rows) > 1 else None
    return dict(place_id=pid.group(1) if pid else None, google_id=gid.group(1) if gid else None,
                name=c.get("name"), latitude=float(lat.group(1)) if lat else None,
                longitude=float(lon.group(1)) if lon else None, rating=rating, n_reviews=n_reviews,
                category=category, address=address, price=price, status=status)


# ---------------------------------------------------------------- grid
def make_grid(points: pd.DataFrame, step_m: float = BASE_STEP_M, pad_cells: int = 0) -> list[tuple]:
    """Tâm các ô lưới step_m (trong UTM 49N) có ≥1 điểm, cộng thêm pad_cells ô kề."""
    xy = geo.to_utm(points)
    ix = np.floor(xy[:, 0] / step_m).astype(int)
    iy = np.floor(xy[:, 1] / step_m).astype(int)
    occupied = set(zip(ix, iy))
    cells = set()
    for (i, j) in occupied:
        for di in range(-pad_cells, pad_cells + 1):
            for dj in range(-pad_cells, pad_cells + 1):
                cells.add((i + di, j + dj))
    from pyproj import Transformer
    inv = Transformer.from_crs(f"EPSG:{geo.UTM_EPSG}", "EPSG:4326", always_xy=True)
    out = []
    for (i, j) in sorted(cells):
        lon, lat = inv.transform((i + 0.5) * step_m, (j + 0.5) * step_m)
        out.append((round(lat, 6), round(lon, 6)))
    return out


def subdivide(lat: float, lon: float, step_m: float) -> list[tuple]:
    """4 tâm ô con của ô (lat, lon) cạnh step_m."""
    d = step_m / 4
    dlat = d / 111_320
    dlon = d / (111_320 * np.cos(np.radians(lat)))
    return [(round(lat + a * dlat, 6), round(lon + b * dlon, 6)) for a in (-1, 1) for b in (-1, 1)]


# ---------------------------------------------------------------- scrape
def _cache_path(keyword: str, lat: float, lon: float, zoom: int) -> Path:
    key = f"{keyword}|{lat:.6f}|{lon:.6f}|{zoom}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return CACHE_DIR / f"{h}.json"


class BlockedError(RuntimeError):
    pass


def search_once(page, keyword: str, lat: float, lon: float, zoom: int,
                max_scrolls: int = 30, pause: float = 1.6) -> dict:
    url = f"https://www.google.com/maps/search/{quote(keyword)}/@{lat},{lon},{zoom}z?hl=vi"
    page.goto(url, timeout=60_000)
    if "/sorry/" in page.url or "consent." in page.url:
        raise BlockedError(page.url)
    try:
        page.wait_for_selector('div[role="feed"], h1', timeout=20_000)
    except Exception:
        if page.locator("form#captcha-form, #recaptcha").count():
            raise BlockedError("captcha")
        return dict(cards=[], reached_end=True, note="no_feed")
    feed = page.locator('div[role="feed"]')
    if not feed.count():  # chỉ 1 kết quả -> Google mở thẳng trang chi tiết
        return dict(cards=[], reached_end=True, note="single_place")
    last, still = 0, 0
    reached_end = False
    for _ in range(max_scrolls):
        feed.evaluate("el => el.scrollTo(0, el.scrollHeight)")
        time.sleep(pause + random.uniform(0, 0.6))
        if page.locator("span.HlvSq").count():
            reached_end = True
            break
        n = page.locator("a.hfpxzc").count()
        still = still + 1 if n == last else 0
        if still >= 3:
            break
        last = n
    cards = page.evaluate(_EXTRACT_JS)
    return dict(cards=cards, reached_end=reached_end, note="")


def run(keywords=KEYWORDS, max_queries: int | None = None, headless: bool = True,
        delay=(2.0, 4.5), step_m: float = BASE_STEP_M):
    from playwright.sync_api import sync_playwright

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stores = pd.read_csv(ROOT / "data" / "processed" / "stores_clean.csv")
    grid = make_grid(stores, step_m)
    # hàng đợi: (keyword, lat, lon, zoom, step_m, depth)
    queue = [(k, la, lo, BASE_ZOOM, step_m, 0) for k in keywords for (la, lo) in grid]
    print(f"Ô lưới: {len(grid)} | từ khóa: {len(keywords)} | truy vấn ban đầu: {len(queue)}")
    done = sum(_cache_path(k, la, lo, z).exists() for (k, la, lo, z, _, _) in queue)
    print(f"Đã có trong cache: {done}")

    n_new = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context(locale="vi-VN", viewport={"width": 1280, "height": 900})
        page = ctx.new_page()
        t0 = time.time()
        while queue:
            k, la, lo, z, st, depth = queue.pop(0)
            cp = _cache_path(k, la, lo, z)
            if cp.exists():
                res = json.loads(cp.read_text(encoding="utf-8"))
            else:
                if max_queries is not None and n_new >= max_queries:
                    print("Đạt max_queries, dừng.")
                    break
                try:
                    res = search_once(page, k, la, lo, z)
                except BlockedError as e:
                    print("!! Bị chặn / CAPTCHA:", e, "-> dừng. Chạy lại sau (cache được giữ).")
                    break
                res.update(keyword=k, lat=la, lon=lo, zoom=z, step_m=st, depth=depth,
                           scraped_at=time.strftime("%Y-%m-%d %H:%M:%S"))
                cp.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
                n_new += 1
                el = time.time() - t0
                print(f"[{n_new}] {k!r} @({la},{lo}) z{z}: {len(res['cards'])} kết quả"
                      f"{' (chạm trần -> chia ô)' if len(res['cards']) >= CAP_HIT and depth < MAX_DEPTH else ''}"
                      f" | {el / n_new:.1f}s/truy vấn | còn {len(queue)}")
                time.sleep(random.uniform(*delay))
            if len(res["cards"]) >= CAP_HIT and depth < MAX_DEPTH:
                queue[0:0] = [(k, a, b, z, st / 2, depth + 1) for (a, b) in subdivide(la, lo, st)]
        browser.close()
    return merge()


# ---------------------------------------------------------------- merge
def merge() -> pd.DataFrame:
    rows = []
    for f in CACHE_DIR.glob("*.json"):
        res = json.loads(f.read_text(encoding="utf-8"))
        for c in res["cards"]:
            r = parse_card(c)
            r["keyword"] = res["keyword"]
            rows.append(r)
    if not rows:
        print("Cache trống.")
        return pd.DataFrame()
    df = pd.DataFrame(rows).dropna(subset=["place_id"])
    kw = df.groupby("place_id")["keyword"].agg(lambda s: "|".join(sorted(set(s))))
    out = (df.drop(columns="keyword").drop_duplicates("place_id")
           .merge(kw.rename("keywords"), left_on="place_id", right_index=True))
    out["n_hits"] = out["place_id"].map(df["place_id"].value_counts())
    path = OUT_DIR / "gmaps_places.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Gộp {len(df)} bản ghi -> {len(out)} địa điểm duy nhất -> {path.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-queries", type=int, default=None, help="số truy vấn MỚI tối đa lần chạy này")
    ap.add_argument("--keywords", nargs="*", default=KEYWORDS)
    ap.add_argument("--step", type=float, default=BASE_STEP_M)
    ap.add_argument("--headful", action="store_true", help="mở cửa sổ trình duyệt để quan sát")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    if a.merge_only:
        merge()
    else:
        run(keywords=a.keywords, max_queries=a.max_queries, headless=not a.headful, step_m=a.step)
