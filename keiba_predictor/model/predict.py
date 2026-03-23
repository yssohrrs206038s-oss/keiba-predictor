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


def format_prediction(result_df: pd.DataFrame, race_name: str = "") -> str:
    """
    予測結果を見やすいテキスト形式で返す。
    期待値スコア・危険な人気馬判定・EV重視買い目を含む。
    """
    # EV・危険フラグがなければここで計算
    if "ev_score" not in result_df.columns:
        result_df = calc_ev_and_flags(result_df)

    sep    = "=" * 58
    header = f"【KEIBA EDGE 予測結果】{race_name}" if race_name else "【KEIBA EDGE 予測結果】"
    lines  = [sep, header, sep]

    # ── 全馬の確率 + 期待値 ────────────────────────────────
    lines.append("\n■ 各馬の3着以内確率 + 期待値(EV)")
    lines.append(
        f"  {'順':>2}  {'馬番':>3}  {'馬名':<14}  {'確率':>6}  {'EV':>6}  {'人気':>3}  {'オッズ':>5}"
    )
    lines.append("  " + "-" * 52)

    for rank, (_, row) in enumerate(result_df.iterrows(), 1):
        num   = str(row.get("horse_number", "-"))
        name  = str(row.get("horse_name",   "-"))[:14]
        prob  = row["prob_top3"] * 100
        pop   = str(int(row["popularity"])) if pd.notna(row.get("popularity")) else "-"
        odds  = row.get("odds", "-")
        ev    = row.get("ev_score")
        ev_str  = f"★{ev:.2f}" if (pd.notna(ev) and ev >= 1.0) else (f" {ev:.2f}" if pd.notna(ev) else "   -  ")
        danger  = " ⚠" if row.get("is_dangerous", False) else "  "
        lines.append(
            f"  {rank:>2}位  {num:>3}番  {name:<14}  {prob:>5.1f}%"
            f"  {ev_str:>6}  {pop:>3}人気  {str(odds):>5}倍{danger}"
        )

    # ── 予想印 ──────────────────────────────────────────────
    lines += ["", "■ 予想印"]

    def _lbl(row: pd.Series) -> str:
        n   = int(row.get("horse_number", 0)) if pd.notna(row.get("horse_number")) else 0
        ev  = row.get("ev_score")
        ev_part = f"  EV={'★' if (pd.notna(ev) and ev >= 1.2 and row['prob_top3'] >= 0.25) else ''}{ev:.2f}" if pd.notna(ev) else ""
        return f"[{n}番] {row.get('horse_name','?')} ({row['prob_top3']*100:.1f}%{ev_part})"

    lines.append(f"  ◎ 本命: {_lbl(result_df.iloc[0])}")
    if len(result_df) >= 2:
        lines.append(f"  ○ 対抗: {_lbl(result_df.iloc[1])}")

    ana: Optional[pd.Series] = None
    try:
        cands = result_df.iloc[2:].copy()
        cands["_o"] = pd.to_numeric(cands["odds"], errors="coerce")
        hi_odds = cands[cands["_o"] >= 10.0]
        ana = hi_odds.iloc[0] if not hi_odds.empty else (result_df.iloc[2] if len(result_df) >= 3 else None)
    except Exception:
        ana = result_df.iloc[2] if len(result_df) >= 3 else None

    if ana is not None:
        lines.append(f"  △ 穴馬: {_lbl(ana)}")

    # ── EV+ 推奨馬 ──────────────────────────────────────────
    ev_plus = result_df[
        (result_df["ev_score"].fillna(0) >= 1.2) &
        (result_df["prob_top3"] >= 0.25)
    ]
    if not ev_plus.empty:
        lines += ["", "■ ★ EV+推奨馬（確率25%以上 かつ 期待値1.2以上 ─ 買う価値あり）"]
        for _, row in ev_plus.iterrows():
            num  = int(row["horse_number"]) if pd.notna(row.get("horse_number")) else 0
            name = str(row.get("horse_name", ""))
            ev   = row["ev_score"]
            prob = row["prob_top3"] * 100
            odds = row.get("odds", "?")
            lines.append(f"  ★ {num}番 {name:<14}  EV={ev:.2f}  ({prob:.1f}% × {odds}倍)")

    # ── 危険な人気馬 ────────────────────────────────────────
    danger_df = result_df[result_df["is_dangerous"]]
    if not danger_df.empty:
        lines += ["", "■ ⚠ 危険な人気馬（買い控え推奨）"]
        for _, row in danger_df.iterrows():
            num     = int(row["horse_number"]) if pd.notna(row.get("horse_number")) else 0
            name    = str(row.get("horse_name", ""))
            pop     = row.get("popularity", "?")
            reasons = row.get("danger_reasons", [])
            lines.append(f"  ⚠  {num}番 {name}（{pop}番人気）")
            for rsn in reasons:
                lines.append(f"       → {rsn}")

    # ── 推奨買い目（EV 上位優先） ───────────────────────────
    lines += ["", "■ 推奨買い目（EV重視）"]

    buy_nums: list[int] = (
        result_df[result_df["ev_score"].notna()]
        .nlargest(3, "ev_score")["horse_number"]
        .dropna().apply(int).tolist()
    )
    if not buy_nums:
        buy_nums = [
            int(r["horse_number"])
            for _, r in result_df.head(3).iterrows()
            if pd.notna(r.get("horse_number"))
        ]

    if len(buy_nums) >= 2:
        pairs = list(combinations(buy_nums, 2))
        lines.append(f"  馬連 / ワイド (EV上位3頭ボックス):")
        for a, b in pairs:
            lines.append(f"    {a}-{b}")
    if len(buy_nums) >= 3:
        lines.append(f"  三連複: {buy_nums[0]}-{buy_nums[1]}-{buy_nums[2]}")

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

    race_name = race_df["race_name"].iloc[0] if "race_name" in race_df.columns else race_id
    print(format_prediction(result, race_name=race_name))
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


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m keiba_predictor.model.predict <race_id>")
        sys.exit(1)
    predict_from_csv(sys.argv[1])
