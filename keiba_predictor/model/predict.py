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
    推奨買い目を2パターン（安定重視 / 期待値重視）で生成して行リストで返す。

    - 確率TOP3 と EV上位3頭が同じ馬番セットなら1パターンのみ表示。
    - result_df は prob_top3 降順でソート済みであること（predict_race の返り値）。
    """
    prob_nums: list[int] = [
        int(r["horse_number"])
        for _, r in result_df.head(3).iterrows()
        if pd.notna(r.get("horse_number"))
    ]
    ev_nums: list[int] = (
        result_df[result_df["ev_score"].notna()]
        .nlargest(3, "ev_score")["horse_number"]
        .dropna().apply(int).tolist()
    ) if "ev_score" in result_df.columns else []
    if not ev_nums:
        ev_nums = prob_nums

    def _combo(nums: list[int]) -> list[str]:
        out: list[str] = []
        if len(nums) >= 2:
            out.append(f"{indent}馬連 / ワイド:")
            for a, b in combinations(nums, 2):
                out.append(f"{indent}  {a}-{b}")
        if len(nums) >= 3:
            out.append(f"{indent}三連複: {nums[0]}-{nums[1]}-{nums[2]}")
        return out

    lines = ["", "■ 推奨買い目"]
    if set(prob_nums) == set(ev_nums):
        lines.append(f"{indent}【安定重視】確率TOP3で堅く")
        lines += _combo(prob_nums)
    else:
        lines.append(f"{indent}【安定重視】確率TOP3で堅く")
        lines += _combo(prob_nums)
        lines.append(f"{indent}【期待値重視】EV上位3頭で配当狙い")
        lines += _combo(ev_nums)
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
) -> str:
    """
    予測結果をコンパクトなテキスト形式で返す（CLI・Discord 共通）。
    TOP5 + EV/危険馬サマリ + 買い目2パターン。

    Args:
        result_df  : predict_race() + calc_ev_and_flags() 済みの DataFrame
        race_name  : 表示用レース名
        ai_comments: {"馬番(str)": "解説テキスト"} — generate_comments() の返り値
    """
    if "ev_score" not in result_df.columns:
        result_df = calc_ev_and_flags(result_df)

    if ai_comments is None:
        ai_comments = {}

    sep    = "─" * 32
    header = f"🏇 【KEIBA EDGE】{race_name}" if race_name else "🏇 【KEIBA EDGE 予測結果】"
    lines  = [sep, header, sep]

    # ── 穴馬インデックス ────────────────────────────────────
    ana_ridx: Optional[int] = None
    try:
        odds_ser = pd.to_numeric(result_df["odds"], errors="coerce")
        cands = result_df.iloc[2:][odds_ser.iloc[2:].fillna(0) >= 10.0]
        if not cands.empty:
            ana_ridx = cands.index[0]
    except Exception:
        pass

    # ── 印割り当て ───────────────────────────────────────────
    rank_marks: dict[int, str] = {}
    hoshi_done = False
    for rank, (ridx, _) in enumerate(result_df.head(5).iterrows()):
        if rank == 0:
            rank_marks[ridx] = "◎"
        elif rank == 1:
            rank_marks[ridx] = "○"
        elif ridx == ana_ridx:
            rank_marks[ridx] = "△"
        elif not hoshi_done:
            rank_marks[ridx] = "☆"
            hoshi_done = True
        else:
            rank_marks[ridx] = " "
    if ana_ridx is not None:
        for rank, (ridx, _) in enumerate(result_df.head(5).iterrows()):
            if rank >= 2 and ridx not in rank_marks:
                if not hoshi_done:
                    rank_marks[ridx] = "☆"
                    hoshi_done = True
                else:
                    rank_marks[ridx] = " "

    # ── TOP5 一行表示 ────────────────────────────────────────
    for _, (ridx, row) in enumerate(result_df.head(5).iterrows()):
        mark  = rank_marks.get(ridx, " ")
        num   = str(int(row["horse_number"])) if pd.notna(row.get("horse_number")) else "-"
        name  = str(row.get("horse_name", "-"))[:10]
        prob  = row["prob_top3"] * 100
        pop   = str(int(row["popularity"])) if pd.notna(row.get("popularity")) else "-"
        odds  = row.get("odds", "-")
        ev    = row.get("ev_score")
        ev_str = f"EV{ev:.2f}" if pd.notna(ev) else "      "
        warn  = " ⚠" if row.get("is_dangerous", False) else ""
        lines.append(
            f"{mark} {num:>2}番 {name:<10} {prob:>5.1f}%  {ev_str}  {pop:>2}人気 {str(odds):>5}倍{warn}"
        )
        comment = ai_comments.get(num, "")
        if comment:
            lines.append(f"  📝 {comment}")

    lines.append(sep)

    # ── ★穴馬注目（TOP5外・EV≥3.0・確率≥15%） ──────────────
    top5_idx = result_df.head(5).index
    ana_hidden = result_df.loc[
        ~result_df.index.isin(top5_idx) &
        (result_df["ev_score"].fillna(0) >= 3.0) &
        (result_df["prob_top3"] >= 0.15)
    ]
    if not ana_hidden.empty:
        row  = ana_hidden.nlargest(1, "ev_score").iloc[0]
        num  = int(row["horse_number"]) if pd.notna(row.get("horse_number")) else 0
        name = str(row.get("horse_name", ""))
        ev   = row["ev_score"]
        pop  = str(int(row["popularity"])) if pd.notna(row.get("popularity")) else "-"
        odds = row.get("odds", "-")
        lines.append(f"★穴馬注目 {num}番{name} EV{ev:.2f}（{pop}人気 {odds}倍）")
        lines.append("")

    # ── ⚠危険な人気馬（1行） ────────────────────────────────
    danger_df = result_df[result_df["is_dangerous"]]
    if not danger_df.empty:
        for _, row in danger_df.iterrows():
            num     = int(row["horse_number"]) if pd.notna(row.get("horse_number")) else 0
            name    = str(row.get("horse_name", ""))
            reasons = row.get("danger_reasons", [])
            short   = reasons[0].split("（")[0] if reasons else "要注意"
            lines.append(f"⚠危険 {num}番{name}（{short}）")
            comment = ai_comments.get(str(num), "")
            if comment:
                lines.append(f"  📝 {comment}")
        lines.append("")

    # ── 推奨買い目（2パターン、1行コンパクト） ───────────────
    prob_nums: list[int] = [
        int(r["horse_number"])
        for _, r in result_df.head(3).iterrows()
        if pd.notna(r.get("horse_number"))
    ]
    ev_nums: list[int] = (
        result_df[result_df["ev_score"].notna()]
        .nlargest(3, "ev_score")["horse_number"]
        .dropna().apply(int).tolist()
    ) if "ev_score" in result_df.columns else []
    if not ev_nums:
        ev_nums = prob_nums

    def _buy_line(nums: list[int]) -> str:
        pairs  = " / ".join(f"{a}-{b}" for a, b in combinations(nums, 2)) if len(nums) >= 2 else ""
        sanren = f"  3連複:{nums[0]}-{nums[1]}-{nums[2]}" if len(nums) >= 3 else ""
        return f"{pairs}{sanren}"

    if set(prob_nums) == set(ev_nums):
        lines.append(f"【安定重視】{_buy_line(prob_nums)}")
    else:
        lines.append(f"【安定重視】{_buy_line(prob_nums)}")
        lines.append(f"【期待値重視】{_buy_line(ev_nums)}")

    lines.append(sep)
    return "\n".join(lines)


def predict_from_csv(
    race_id: str,
    featured_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    featured_races.csv から指定 race_id のレースを抽出して予測する。

    Args:
        race_id:       予測対象のレースID
        featured_path: 特徴量付きCSVのパス
        model_path:    モデルファイルパス

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

    race_name = race_df["race_name"].iloc[0] if "race_name" in race_df.columns else race_id
    from keiba_predictor.ai_comment import generate_comments
    ai_comments = generate_comments(result, race_name=race_name)
    print(format_prediction(result, race_name=race_name, ai_comments=ai_comments))
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
    print(format_prediction(result, race_name=race_name))
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
    msg = format_prediction(result, race_name=f"{race_name}  {course_info}", ai_comments=ai_comments)
    print(msg)

    if notify:
        import os
        from keiba_predictor.discord_notify import send_discord
        url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")
        if not url:
            logger.error("--webhook-url または環境変数 DISCORD_WEBHOOK_URL を指定してください")
        else:
            ok = send_discord(url, f"```\n{msg}\n```")
            logger.info("Discord 送信完了" if ok else "Discord 送信失敗")

    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m keiba_predictor.model.predict <race_id>")
        sys.exit(1)
    predict_from_csv(sys.argv[1])
