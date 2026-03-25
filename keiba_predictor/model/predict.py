"""
予測モジュール

- 学習済みモデルを使って指定レースの3着以内確率を出力
- 本命・対抗・穴馬の推奨
- 期待値スコア（EV）・危険な人気馬判定
- 馬連・ワイドの推奨組み合わせ
"""

import pickle
import logging
import re
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from keiba_predictor.features.feature_engineering import FEATURE_COLS

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = Path(__file__).parent / "xgb_model.pkl"

# JRA 競馬場コード → 競馬場名
VENUE_MAP: dict[str, str] = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


def _build_course_info(race_id: str, race_df: pd.DataFrame) -> str:
    """race_id と DataFrame からコース情報文字列（例: 小倉 芝1800m）を組み立てる。"""
    venue = VENUE_MAP.get(str(race_id)[4:6], "")
    ct_str = ""
    if "course_type" in race_df.columns and "distance" in race_df.columns:
        ct  = race_df["course_type"].iloc[0]
        dst = race_df["distance"].iloc[0]
        if pd.notna(ct) and pd.notna(dst):
            ct_str = f"{ct}{int(dst)}m"
    if venue and ct_str:
        return f"{venue} {ct_str}"
    return venue or ct_str


def calc_ev_and_flags(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    result_df に期待値スコアと危険フラグを付与して返す。

    追加列:
      ev_score       : float  ─ prob_top3 × odds
      is_dangerous   : bool
      danger_reasons : list[str]  ─ 危険と判断した理由

    危険馬の条件（5番人気以内の馬が対象）:
      1. AI 3着以内確率 < 40% かつ 3番人気以内
      2. 1〜2番人気 かつ 前走5着以下
    """
    df = result_df.copy()

    # 期待値: 3着以内確率 × 複勝オッズ（単勝オッズで近似）
    odds_num = pd.to_numeric(df["odds"], errors="coerce")
    df["ev_score"] = df["prob_top3"] * odds_num

    def _reasons(row: pd.Series) -> list[str]:
        pop   = pd.to_numeric(row.get("popularity"),      errors="coerce")
        pfp   = pd.to_numeric(row.get("prev_finish_pos"), errors="coerce")
        prob  = float(row["prob_top3"])
        out: list[str] = []
        if pd.notna(pop) and pop <= 3 and prob < 0.40:
            out.append(f"AI確率{prob*100:.0f}%（3番人気以内なのに低い）")
        if pd.notna(pop) and pop <= 2 and pd.notna(pfp) and pfp >= 5:
            out.append(f"1〜2番人気だが前走{int(pfp)}着")
        return out

    df["danger_reasons"] = df.apply(_reasons, axis=1)
    df["is_dangerous"]   = df["danger_reasons"].apply(bool)
    return df


def format_buy_patterns(result_df: pd.DataFrame, indent: str = "  ") -> list[str]:
    """
    推奨買い目を生成して行リストで返す。

    - 複勝: ◎本命1頭（1点）
    - 馬連: ◎ → ○☆△への流し（3点）
    - 3連複: ◎軸 × ○☆△4番手5番手の5頭流し（10点）
    合計14点
    """
    top6 = result_df.head(6)
    nums = [
        int(r["horse_number"])
        for _, r in top6.iterrows()
        if pd.notna(r.get("horse_number"))
    ]
    if len(nums) < 2:
        return []

    axis      = nums[0]
    umaren    = nums[1:4]           # ○☆△ (3頭)
    sanren    = nums[1:6]           # ○☆△4番手5番手 (最大5頭)

    fukusho_name = str(top6.iloc[0].get("horse_name", "")) if len(top6) >= 1 else ""
    umaren_str   = " / ".join(f"{axis}-{n}" for n in umaren) if umaren else ""
    sanren_str   = "/".join(str(n) for n in sanren)

    lines = [
        "",
        f"■ 推奨買い目（14点）",
        f"{indent}複勝:  {axis}番（{fukusho_name}）",
        f"{indent}馬連:  {umaren_str}",
        f"{indent}3連複: 軸{axis}番 × {sanren_str}",
    ]
    return lines


def load_model(model_path: Path | None = None) -> dict:
    """学習済みモデルをロードする。"""
    if model_path is None:
        model_path = MODEL_PATH
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    logger.info(
        f"モデルロード完了 | 学習時AUC: {bundle.get('cv_auc_mean', 'N/A'):.4f}, "
        f"複勝的中率: {bundle.get('cv_fukusho_mean', 'N/A'):.4f}"
    )
    return bundle


def predict_race(
    race_df: pd.DataFrame,
    model_bundle: Optional[dict] = None,
    model_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    1レース分のDataFrameに対して3着以内確率を予測する。

    Args:
        race_df:      1レース分のデータ（feature_engineering済み）
        model_bundle: load_model() の返り値（省略時は自動ロード）
        model_path:   モデルファイルパス（省略時はデフォルト）

    Returns:
        確率列 'prob_top3' を追加したDataFrame（降順ソート済み）
    """
    if model_bundle is None:
        model_bundle = load_model(model_path)

    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]

    # 特徴量が存在しない列は NaN で補完
    for col in feature_cols:
        if col not in race_df.columns:
            race_df[col] = np.nan

    X = race_df[feature_cols].astype(float)
    probs = model.predict_proba(X)[:, 1]

    result = race_df.copy()
    result["prob_top3"] = probs
    result = result.sort_values("prob_top3", ascending=False).reset_index(drop=True)
    return result


def format_prediction(
    result_df: pd.DataFrame,
    race_name: str = "",
    ai_comments: Optional[dict] = None,
    course_info: str = "",
) -> tuple[str, str]:
    """
    予測結果を競馬新聞風の2メッセージ（予想・買い目）で返す。

    Args:
        result_df  : predict_race() + calc_ev_and_flags() 済みの DataFrame
        race_name  : 表示用レース名
        ai_comments: {"馬番(str)": "解説テキスト"} — generate_comments() の返り値
        course_info: コース情報（例: "芝2000m"）

    Returns:
        (msg1_予想, msg2_買い目) のタプル
    """
    if "ev_score" not in result_df.columns:
        result_df = calc_ev_and_flags(result_df)

    if ai_comments is None:
        ai_comments = {}

    sep = "━" * 20

    # ── Message 1: 予想 ───────────────────────────────────────
    race_label = race_name if race_name else "KEIBA EDGE 予測結果"
    lines1 = [sep, f"🏇 {race_label}"]
    if course_info:
        lines1.append(course_info)
    lines1.append(sep)

    # 印: rank 0=◎, 1=○, 2=☆, 3=△, 4=空白
    MARKS = ["◎", "○", "☆", "△", "　"]

    top5 = result_df.head(5)
    top5_idx = top5.index

    for rank, (ridx, row) in enumerate(top5.iterrows()):
        mark     = MARKS[rank] if rank < len(MARKS) else "　"
        num      = str(int(row["horse_number"])) if pd.notna(row.get("horse_number")) else "-"
        name     = str(row.get("horse_name", "-"))
        prob     = row["prob_top3"] * 100
        pop      = str(int(row["popularity"])) if pd.notna(row.get("popularity")) else "-"
        odds_val = row.get("odds", "-")
        ev       = row.get("ev_score")
        ev_str   = f"EV{ev:.2f}" if pd.notna(ev) else ""
        pfp      = row.get("prev_finish_pos")
        pfp_str  = f"前走{int(pfp)}着" if pd.notna(pfp) and float(pfp) > 0 else ""

        # 馬名行
        lines1.append(f"{mark} {num}番 {name}")

        # 人気・オッズ・前走着順
        stat = f"　{pop}人気 {odds_val}倍"
        if pfp_str:
            stat += f" | {pfp_str}"
        lines1.append(stat)

        # AI確率・EV
        prob_line = f"　AI確率{prob:.1f}%"
        if ev_str:
            prob_line += f" {ev_str}"
        lines1.append(prob_line)

        # AI解説（◎○☆のみ）
        if mark in ("◎", "○", "☆"):
            comment = ai_comments.get(num, "")
            if comment:
                lines1.append(f"　📝 {comment}")

    lines1.append(sep)

    # ★穴馬（TOP5外・EV≥3.0・確率≥15%）
    ana_hidden = result_df.loc[
        ~result_df.index.isin(top5_idx) &
        (result_df["ev_score"].fillna(0) >= 3.0) &
        (result_df["prob_top3"] >= 0.15)
    ]
    if not ana_hidden.empty:
        row      = ana_hidden.nlargest(1, "ev_score").iloc[0]
        num      = int(row["horse_number"]) if pd.notna(row.get("horse_number")) else 0
        name     = str(row.get("horse_name", ""))
        ev       = row["ev_score"]
        pop      = str(int(row["popularity"])) if pd.notna(row.get("popularity")) else "-"
        odds_val = row.get("odds", "-")
        lines1.append(f"★穴馬 {num}番{name} EV{ev:.2f}（{pop}人気 {odds_val}倍）")

    # ⚠危険な人気馬
    danger_df = result_df[result_df["is_dangerous"]]
    if not danger_df.empty:
        for _, row in danger_df.iterrows():
            num     = int(row["horse_number"]) if pd.notna(row.get("horse_number")) else 0
            name    = str(row.get("horse_name", ""))
            reasons = row.get("danger_reasons", [])
            reason  = reasons[0] if reasons else "要注意"
            lines1.append(f"⚠危険 {num}番{name}（{reason}）")

    lines1.append(sep)
    msg1 = "\n".join(lines1)

    # ── Message 2: 買い目 ────────────────────────────────────
    buy_header = f"💰 {race_name}  買い目" if race_name else "💰 買い目"
    lines2 = [sep, buy_header, sep]

    top6     = result_df.head(6)
    buy_nums = [
        int(r["horse_number"])
        for _, r in top6.iterrows()
        if pd.notna(r.get("horse_number"))
    ]

    if len(buy_nums) >= 2:
        axis        = buy_nums[0]
        axis_name   = str(top6.iloc[0].get("horse_name", "")) if len(top6) >= 1 else ""
        umaren_flow = buy_nums[1:4]
        sanren_flow = buy_nums[1:6]
        umaren_str  = " / ".join(f"{axis}-{n}" for n in umaren_flow)
        sanren_str  = "/".join(str(n) for n in sanren_flow)

        lines2 += [
            "■ 複勝（1点）",
            f"　{axis}番 {axis_name}",
            "■ 馬連（3点）",
            f"　{umaren_str}",
            "■ 3連複（10点）",
            f"　軸 {axis}番",
            f"　× {sanren_str}",
            sep,
            "合計 14点",
        ]

    lines2.append(sep)
    msg2 = "\n".join(lines2)

    return msg1, msg2


def predict_from_csv(
    race_id: str,
    featured_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    notify: bool = False,
    webhook_url: Optional[str] = None,
) -> pd.DataFrame:
    """
    featured_races.csv から指定 race_id のレースを抽出して予測する。

    Args:
        race_id:       予測対象のレースID
        featured_path: 特徴量付きCSVのパス
        model_path:    モデルファイルパス
        notify:        True のとき Discord に予測結果を送信
        webhook_url:   Discord Webhook URL（notify=True 時に使用）

    Returns:
        予測結果DataFrame
    """
    if featured_path is None:
        featured_path = DATA_DIR / "featured_races.csv"

    df = pd.read_csv(featured_path, encoding="utf-8-sig", parse_dates=["race_date"])
    race_df = df[df["race_id"].astype(str) == str(race_id)].copy()

    if race_df.empty:
        raise ValueError(f"race_id={race_id} がデータに存在しません")

    model_bundle = load_model(model_path)
    result = predict_race(race_df, model_bundle)
    result = calc_ev_and_flags(result)

    race_name   = race_df["race_name"].iloc[0] if "race_name" in race_df.columns else race_id
    course_info = _build_course_info(race_id, race_df)
    print(f"[DEBUG] race_id={race_id}  venue_code={str(race_id)[4:6]!r}  course_info={course_info!r}", flush=True)

    from keiba_predictor.ai_comment import generate_comments
    ai_comments = generate_comments(result, race_name=race_name, course_info=course_info)
    print(f"[DEBUG] ai_comments: {len(ai_comments)} 頭分  keys={sorted(ai_comments.keys())}", flush=True)

    msg1, msg2 = format_prediction(result, race_name=race_name, ai_comments=ai_comments,
                                   course_info=course_info)
    print(msg1)
    print(msg2)

    # 予想キャッシュに保存（note_report・結果照合で使用）
    race_date = ""
    if "race_date" in race_df.columns:
        try:
            race_date = str(race_df["race_date"].iloc[0].date())
        except Exception:
            race_date = str(race_df["race_date"].iloc[0])
    from keiba_predictor.discord_notify import _store_prediction
    _store_prediction(race_id, race_name, race_date, result,
                      ai_comments=ai_comments, course_info=course_info)

    if notify:
        import os
        from keiba_predictor.discord_notify import send_discord
        url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")
        if not url:
            logger.error("--webhook-url または環境変数 DISCORD_WEBHOOK_URL を指定してください")
        else:
            ok = send_discord(url, msg1) and send_discord(url, msg2)
            logger.info(f"Discord 送信{'完了' if ok else '失敗'}")

    return result


def predict_upcoming(
    race_df: pd.DataFrame,
    race_name: str = "",
    model_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    未来のレース（スクレイピングしてきたデータ）に対して予測する。
    feature_engineering を経たDataFrameを渡すこと。

    Args:
        race_df:    特徴量付きの1レース分DataFrame
        race_name:  表示用レース名
        model_path: モデルファイルパス

    Returns:
        予測結果DataFrame
    """
    model_bundle = load_model(model_path)
    result = predict_race(race_df, model_bundle)
    msg1, msg2 = format_prediction(result, race_name=race_name)
    print(msg1)
    print(msg2)
    return result


def predict_live(
    race_id: str,
    notify: bool = False,
    webhook_url: Optional[str] = None,
    model_path: Optional[Path] = None,
    cleaned_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    出馬表をリアルタイムでスクレイピングして予測する。

    過去CSVにないレースや未来レースでも利用可能。
    過去成績がない馬はデータセット中央値で補完する。

    Args:
        race_id      : netkeibaのレースID
        notify       : True のとき Discord に予測結果を送信
        webhook_url  : Discord Webhook URL（notify=True 時に使用）
        model_path   : モデルファイルパス
        cleaned_path : 過去成績クリーニング済みCSVのパス

    Returns:
        予測結果DataFrame
    """
    from keiba_predictor.scraper.shutuba_scraper import scrape_shutuba
    from keiba_predictor.features.live_features import build_live_features

    # 出馬表を取得
    shutuba_info = scrape_shutuba(race_id)
    if shutuba_info is None:
        raise ValueError(f"出馬表の取得に失敗しました: race_id={race_id}")

    horses_df = shutuba_info["horses"]
    if horses_df.empty:
        raise ValueError(f"出馬表に馬が見つかりませんでした: race_id={race_id}")

    # 特徴量を生成
    race_df = build_live_features(shutuba_info, cleaned_path=cleaned_path)
    if race_df.empty:
        raise ValueError("特徴量の生成に失敗しました")

    # 予測
    model_bundle = load_model(model_path)
    result = predict_race(race_df, model_bundle)
    result = calc_ev_and_flags(result)

    race_name   = shutuba_info["race_name"]
    course_info = shutuba_info["course_info"]

    from keiba_predictor.ai_comment import generate_comments
    ai_comments = generate_comments(result, race_name=race_name, course_info=course_info)

    msg1, msg2 = format_prediction(result, race_name=race_name, ai_comments=ai_comments, course_info=course_info)
    print(msg1)
    print(msg2)

    # 予想キャッシュに保存（note_report・結果照合で使用）
    race_date = shutuba_info.get("race_date", "")
    from keiba_predictor.discord_notify import _store_prediction
    _store_prediction(race_id, race_name, race_date, result,
                      ai_comments=ai_comments, course_info=course_info)

    if notify:
        import os
        from keiba_predictor.discord_notify import send_discord
        url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")
        if not url:
            logger.error("--webhook-url または環境変数 DISCORD_WEBHOOK_URL を指定してください")
        else:
            ok = send_discord(url, msg1) and send_discord(url, msg2)
            logger.info(f"Discord 送信{'完了' if ok else '失敗'}")

    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m keiba_predictor.model.predict <race_id>")
        sys.exit(1)
    predict_from_csv(sys.argv[1])
