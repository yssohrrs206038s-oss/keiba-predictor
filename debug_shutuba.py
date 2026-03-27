"""
出馬表HTML構造デバッグスクリプト

使い方:
    python debug_shutuba.py 202606030111

取得したHTMLを keiba_predictor/data/debug/shutuba_<race_id>.html に保存し、
テーブル・行・tdのクラス名を標準出力に出力する。
"""

import sys
import re
import pathlib
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main(race_id: str) -> None:
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    print(f"対象URL: {url}")

    # ── HTML取得 (Playwright優先) ──────────────────────────────────
    html = None
    try:
        from playwright.sync_api import sync_playwright
        import time
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                locale="ja-JP",
            )
            page = ctx.new_page()
            page.goto("https://race.netkeiba.com/", wait_until="domcontentloaded", timeout=20000)
            time.sleep(1)
            page.goto(url, wait_until="networkidle", timeout=30000)
            try:
                page.wait_for_selector(
                    "table.Shutuba_Table, tr.HorseList, td.Umaban",
                    timeout=10000,
                )
            except Exception:
                pass
            html = page.content()
            browser.close()
            print(f"Playwright 取得成功: {len(html)} bytes")
    except ImportError:
        print("playwright 未インストール → requests にフォールバック")
    except Exception as e:
        print(f"Playwright 失敗: {e} → requests にフォールバック")

    if html is None:
        import requests, time
        HEADERS = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "ja,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        session = requests.Session()
        session.headers.update(HEADERS)
        try:
            session.get("https://race.netkeiba.com/", timeout=15)
            time.sleep(1)
        except Exception:
            pass
        try:
            r = session.get(url, timeout=15)
            html = r.text
            print(f"requests 取得成功: {len(html)} bytes / status={r.status_code}")
        except Exception as e:
            print(f"requests 取得失敗: {e}")
            sys.exit(1)

    # ── HTMLをファイル保存 ─────────────────────────────────────────
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    debug_dir = pathlib.Path("keiba_predictor/data/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / f"shutuba_{race_id}.html"
    out_path.write_text(soup.prettify(), encoding="utf-8")
    print(f"\nHTML保存: {out_path} ({out_path.stat().st_size} bytes)")

    # ── テーブル一覧 ──────────────────────────────────────────────
    print("\n=== ページ内テーブル ===")
    for i, t in enumerate(soup.find_all("table")):
        print(f"  table[{i}] class={t.get('class')} id={t.get('id')}")

    # ── Shutuba_Table 検索 ────────────────────────────────────────
    print("\n=== Shutuba_Table セレクター確認 ===")
    selectors = [
        "table.Shutuba_Table",
        "table#shutuba_table",
        "table.ShutubaTable",
        "table[class*='Shutuba']",
    ]
    table = None
    for sel in selectors:
        found = soup.select_one(sel)
        print(f"  {sel!r:40s} → {'HIT' if found else 'miss'}")
        if found and table is None:
            table = found

    # ── HorseList 行 ─────────────────────────────────────────────
    print("\n=== HorseList 行セレクター確認 ===")
    row_selectors = [
        "tr.HorseList",
        "tr[class*='HorseList']",
    ]
    for sel in row_selectors:
        trs = (table or soup).select(sel)
        print(f"  {sel!r:40s} → {len(trs)} 行")

    # Umaban を持つ行
    if table:
        umaban_trs = [tr for tr in table.find_all("tr") if tr.select_one("td.Umaban")]
        print(f"  {'td.Umaban を持つ tr':40s} → {len(umaban_trs)} 行")

    # ── 最初の1行のtd構造ダンプ ───────────────────────────────────
    print("\n=== 最初の馬行の td 構造 ===")
    target_trs = []
    if table:
        target_trs = table.select("tr.HorseList, tr[class*='HorseList']")
        if not target_trs:
            target_trs = [tr for tr in table.find_all("tr") if tr.select_one("td.Umaban")]
    if not target_trs:
        target_trs = soup.select("tr.HorseList, tr[class*='HorseList']")

    if target_trs:
        first_tr = target_trs[0]
        print(f"  tr.class = {first_tr.get('class')}")
        for j, td in enumerate(first_tr.find_all("td")):
            text = td.get_text(strip=True)[:40]
            print(f"  td[{j:02d}] class={td.get('class')} text={text!r}")
    else:
        print("  行データが見つかりませんでした")

    # ── 個別フィールドのセレクター確認 ───────────────────────────
    if target_trs:
        print("\n=== 個別フィールドのセレクター確認（1行目）===")
        tr = target_trs[0]
        checks = [
            (".Umaban", "td.Umaban"),
            (".Waku", "td.Waku"),
            (".HorseName a", "td.HorseInfo a"),
            (".Barei", "td.Barei", "td.sexage"),
            (".Futan", "td.Futan", "td.Wt"),
            (".HorseWeight", "td.HorseWeight", "td.Weight"),
            (".Jockey a", "td.Jockey a"),
            (".Trainer a", "td.Trainer a"),
            (".Odds", "td.Odds"),
            (".Popular", "td.Popular", "td.popular_rank"),
        ]
        for sels in checks:
            label = sels[0]
            found_text = ""
            found_sel = ""
            for sel in sels:
                el = tr.select_one(sel)
                if el:
                    found_text = el.get_text(strip=True)[:30]
                    found_sel = sel
                    break
            status = f"HIT ({found_sel!r}) = {found_text!r}" if found_sel else "miss"
            print(f"  {label!r:25s} → {status}")

    print(f"\n完了。HTMLの詳細は {out_path} を参照してください。")


if __name__ == "__main__":
    race_id = sys.argv[1] if len(sys.argv) > 1 else "202606030111"
    main(race_id)
