"""
週末重賞自動予想 → Discord通知スクリプト

【使い方】
    python -m keiba_predictor.discord_notify

【環境変数】
    DISCORD_WEBHOOK_URL : Discord の Incoming Webhook URL
    または --webhook-url オプションで直接指定可能

【前提条件】
    1. モデルが学習済み (keiba_predictor/model/xgb_model.pkl)
    2. 特徴量CSVが生成済み (keiba_predictor/data/featured_races.csv)
    上記がない場合は先に以下を実行してください:
        python -m keiba_predictor.main all --start 2023-01 --end YYYY-MM

【Windowsタスクスケジューラ】
    keiba_notify.bat をプロジェクトルートに配置し、
    毎週金曜 09:00 に実行登録してください（README参照）。
"""

import argparse
import logging
import os
import re
import sys
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

from keiba_predictor.scraper.netkeiba_scraper import (
    _get, _sleep, RACE_LIST_URL,
)
from keiba_predictor.model.predict import load_model, predict_race

logger = logging.getLogger(__name__)

DATA_DIR  = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "model"

# 重賞判定パターン: "(G1)" "(GI)" "(GII)" "(GIII)" 等
GRADE_RE = re.compile(r"\(G[Ⅰ-Ⅲ1-3]\)|\(GI{1,3}\)")

MARK_HONMEI = "◎"
MARK_TAIKOU = "○"
MARK_ANA    = "△"


# ── Discord送信 ──────────────────────────────────────────────────

def send_discord(webhook_url: str, content: str) -> bool:
    """Discord Webhook にメッセージを送信する。2000字超は自動分割。"""
    if not webhook_url:
        logger.error("Discord Webhook URL が未設定です")
        return False

    # Discord の 1メッセージ上限は 2000 文字
    chunks = [content[i : i + 1900] for i in range(0, len(content), 1900)]
    success = True
    for chunk in chunks:
        try:
            resp = requests.post(
                webhook_url, json={"content": chunk}, timeout=15
            )
            if resp.status_code not in (200, 204):
                logger.error(f"Discord送信失敗: {resp.status_code} {resp.text[:200]}")
                success = False
        except requests.RequestException as e:
            logger.error(f"Discord送信エラー: {e}")
            success = False
    return success


# ── 週末重賞レース取得 ───────────────────────────────────────────

def _weekend_kaisai_dates() -> list[str]:
    """今週末（土・日）の開催日を YYYYMMDD 形式で返す。

    金曜実行を想定: 翌日=土、翌々日=日。
    土曜実行時は当日・翌日、それ以外は直近の土・日を返す。
    """
    today   = date.today()
    weekday = today.weekday()  # 0=月 … 5=土 6=日

    if weekday == 4:      # 金曜 → 明日が土
        days_to_sat = 1
    elif weekday == 5:    # 土曜 → 当日
        days_to_sat = 0
    elif weekday == 6:    # 日曜 → 昨日が土
        days_to_sat = -1
    else:                 # 月〜木 → 次の土曜
        days_to_sat = 5 - weekday

    saturday = today + timedelta(days=days_to_sat)
    sunday   = saturday + timedelta(days=1)
    return [saturday.strftime("%Y%m%d"), sunday.strftime("%Y%m%d")]


def scrape_grade_race_ids(session: requests.Session) -> list[dict]:
    """今週末の重賞レース一覧を取得する。

    Returns:
        [{"race_id": "202603220911", "race_name": "...(G1)", "race_date": "2026-03-22"}, ...]
    """
    found: list[dict] = []
    seen:  set[str]   = set()

    for kaisai_date in _weekend_kaisai_dates():
        url  = f"{RACE_LIST_URL}?kaisai_date={kaisai_date}"
        soup = _get(url, session)
        if soup is None:
            logger.warning(f"  race_list_sub 取得失敗: {kaisai_date}")
            continue

        for a in soup.select("a[href*='race_id=']"):
            href = a.get("href", "")
            m = re.search(r"race_id=(\d{12})", href)
            if not m:
                continue
            race_id = m.group(1)
            if race_id in seen:
                continue

            # レース名
            name_el  = (
                a.select_one(".RaceName")
                or a.select_one(".RaceList_ItemTitle")
                or a.select_one("span")
            )
            race_name = (
                name_el.get_text(strip=True) if name_el else a.get_text(strip=True)
            )

            # 重賞判定
            is_grade = bool(GRADE_RE.search(race_name))
            if not is_grade:
                cls_str  = " ".join(a.get("class", []))
                is_grade = "isGrade" in cls_str or "Grade" in cls_str

            if is_grade:
                seen.add(race_id)
                found.append({
                    "race_id":   race_id,
                    "race_name": race_name,
                    "race_date": f"{kaisai_date[:4]}-{kaisai_date[4:6]}-{kaisai_date[6:8]}",
                })
                logger.info(f"  重賞発見: {race_name} ({race_id})")

        _sleep()

    return found


# ── 予想フォーマット ─────────────────────────────────────────────

def _format_discord_message(
    result_df: pd.DataFrame,
    race_name: str,
    race_date: str,
) -> str:
    """Discord 向け予想メッセージを生成する（コードブロック形式）。"""
    lines = [f"```\n🏇 {race_name}  {race_date}", "=" * 46]

    # ── TOP5 確率 ─────────────────────────────────────────
    lines.append("■ 3着以内確率 TOP5")
    lines.append(f"  {'印':1}  {'馬番':>3}  {'馬名':<12}  {'確率':>6}  {'人気':>3}")
    lines.append("  " + "-" * 38)

    # 穴馬インデックス (3位以降でオッズ10倍以上の最高確率馬)
    ana_ridx: Optional[int] = None
    try:
        odds_ser = pd.to_numeric(result_df["odds"], errors="coerce")
        cands = result_df.iloc[2:][odds_ser.iloc[2:].fillna(0) >= 10.0]
        if not cands.empty:
            ana_ridx = cands.index[0]
    except Exception:
        pass

    mark_map = {0: MARK_HONMEI, 1: MARK_TAIKOU}

    for rank, (ridx, row) in enumerate(result_df.head(5).iterrows()):
        mark = mark_map.get(rank, "")
        if ridx == ana_ridx:
            mark = MARK_ANA
        num  = str(int(row.get("horse_number", 0))) if pd.notna(row.get("horse_number")) else "-"
        name = str(row.get("horse_name", "-"))[:12]
        prob = row["prob_top3"] * 100
        pop  = str(int(row["popularity"])) if pd.notna(row.get("popularity")) else "-"
        lines.append(f"  {mark or ' ':1}  {num:>3}番  {name:<12}  {prob:>5.1f}%  {pop}人気")

    # ── 予想印 ────────────────────────────────────────────
    lines.append("")
    lines.append("■ 予想印")
    honmei = result_df.iloc[0]
    taikou = result_df.iloc[1] if len(result_df) >= 2 else None

    def _label(row: pd.Series) -> str:
        num  = int(row.get("horse_number", 0)) if pd.notna(row.get("horse_number")) else 0
        name = str(row.get("horse_name", "?"))
        prob = row["prob_top3"] * 100
        return f"{num}番 {name} ({prob:.1f}%)"

    lines.append(f"  {MARK_HONMEI} 本命: {_label(honmei)}")
    if taikou is not None:
        lines.append(f"  {MARK_TAIKOU} 対抗: {_label(taikou)}")

    if ana_ridx is not None:
        ana = result_df.loc[ana_ridx]
    elif len(result_df) >= 3:
        ana = result_df.iloc[2]
    else:
        ana = None
    if ana is not None:
        lines.append(f"  {MARK_ANA} 穴馬: {_label(ana)}")

    # ── 推奨買い目 ───────────────────────────────────────
    lines.append("")
    lines.append("■ 推奨買い目")

    top3_nums = []
    for _, row in result_df.head(3).iterrows():
        v = row.get("horse_number")
        if pd.notna(v):
            top3_nums.append(int(v))

    if len(top3_nums) >= 2:
        pairs = list(combinations(top3_nums, 2))
        umaren_str = " / ".join(f"{a}-{b}" for a, b in pairs)
        lines.append(f"  馬連  : {umaren_str}")
        lines.append(f"  ワイド: {umaren_str}")
    if len(top3_nums) >= 3:
        lines.append(f"  三連複: {top3_nums[0]}-{top3_nums[1]}-{top3_nums[2]}")

    lines.append("```")
    return "\n".join(lines)


# ── メイン処理 ───────────────────────────────────────────────────

def run_weekly_notify(
    webhook_url: Optional[str] = None,
    featured_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
) -> None:
    """週末重賞を予想して Discord に通知する。

    毎週金曜 09:00 に Windows タスクスケジューラ等から呼び出す想定。
    """
    # --- Webhook URL ---
    if not webhook_url:
        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        raise ValueError(
            "Discord Webhook URL が未設定です。\n"
            "環境変数 DISCORD_WEBHOOK_URL を設定するか "
            "--webhook-url オプションを指定してください。"
        )

    # --- パス解決 ---
    if featured_path is None:
        featured_path = DATA_DIR / "featured_races.csv"
    if model_path is None:
        model_path = MODEL_DIR / "xgb_model.pkl"

    session = requests.Session()

    # ── 重賞レース取得 ──────────────────────────────────
    logger.info("週末重賞レースを検索中...")
    grade_races = scrape_grade_race_ids(session)

    if not grade_races:
        send_discord(webhook_url, "🏇 今週末の重賞レース情報が取得できませんでした。")
        return

    # ── 前提ファイル確認 ────────────────────────────────
    if not featured_path.exists():
        msg = (
            "⚠️ 特徴量データが見つかりません。\n"
            "先に以下を実行してください:\n"
            "```\npython -m keiba_predictor.main all --start 2023-01 --end YYYY-MM\n```"
        )
        send_discord(webhook_url, msg)
        return

    if not model_path.exists():
        msg = (
            "⚠️ モデルファイルが見つかりません。\n"
            "先に以下を実行してください:\n"
            "```\npython -m keiba_predictor.main train\n```"
        )
        send_discord(webhook_url, msg)
        return

    # ── モデル・データ読み込み ────────────────────────
    model_bundle = load_model(model_path)
    df_all = pd.read_csv(
        featured_path, encoding="utf-8-sig", parse_dates=["race_date"]
    )

    # ── ヘッダー送信 ─────────────────────────────────
    dates_str = " / ".join(sorted({r["race_date"] for r in grade_races}))
    header = (
        f"🏇 **今週末の重賞予想** ({dates_str})\n"
        f"対象: {len(grade_races)} レース"
    )
    send_discord(webhook_url, header)

    # ── 各重賞を予測して送信 ──────────────────────────
    notified = 0
    skipped  = 0
    for race in grade_races:
        race_id   = race["race_id"]
        race_name = race["race_name"]
        race_date = race["race_date"]

        race_df = df_all[df_all["race_id"].astype(str) == str(race_id)].copy()

        if race_df.empty:
            logger.warning(f"  データなし: {race_name} ({race_id})")
            send_discord(
                webhook_url,
                f"⚠️ **{race_name}** のデータが見つかりません (race_id: `{race_id}`)\n"
                "スクレイピング後に再実行してください。",
            )
            skipped += 1
            continue

        result = predict_race(race_df, model_bundle)
        msg    = _format_discord_message(result, race_name, race_date)

        if send_discord(webhook_url, msg):
            notified += 1
            logger.info(f"  送信完了: {race_name}")
        else:
            logger.error(f"  送信失敗: {race_name}")

    # ── フッター ─────────────────────────────────────
    send_discord(
        webhook_url,
        f"✅ {notified}/{len(grade_races)} レース送信完了"
        + (f"（{skipped} レースはデータなし）" if skipped else ""),
    )


# ── エントリポイント ─────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="週末重賞の予想を Discord に送信する",
    )
    parser.add_argument(
        "--webhook-url",
        help="Discord Webhook URL（未指定時は環境変数 DISCORD_WEBHOOK_URL を使用）",
    )
    parser.add_argument(
        "--featured-csv",
        help="特徴量CSVパス（デフォルト: keiba_predictor/data/featured_races.csv）",
    )
    parser.add_argument(
        "--model",
        help="モデルファイルパス（デフォルト: keiba_predictor/model/xgb_model.pkl）",
    )
    args = parser.parse_args()

    featured_path = Path(args.featured_csv) if args.featured_csv else None
    model_path    = Path(args.model)         if args.model         else None

    run_weekly_notify(
        webhook_url   = args.webhook_url,
        featured_path = featured_path,
        model_path    = model_path,
    )


if __name__ == "__main__":
    main()
