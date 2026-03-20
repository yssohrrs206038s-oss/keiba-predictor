"""
予測モジュール

- 学習済みモデルを使って指定レースの3着以内確率を出力
- 本命・対抗・穴馬の推奨
- 馬連・ワイドの推奨組み合わせ
"""

import pickle
import logging
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from keiba_predictor.features.feature_engineering import FEATURE_COLS

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = Path(__file__).parent / "xgb_model.pkl"


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
    本命・対抗・穴馬の推奨と馬連・ワイドの組み合わせも含む。
    """
    lines = []
    sep = "=" * 55

    header = f"【予測結果】{race_name}" if race_name else "【予測結果】"
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    # ── 全馬の確率一覧 ─────────────────────────────────────
    lines.append("\n■ 各馬の3着以内確率")
    lines.append(f"{'順位':>4}  {'馬番':>3}  {'馬名':<16}  {'確率':>7}  {'人気':>3}  {'オッズ':>6}")
    lines.append("-" * 50)

    for rank, (_, row) in enumerate(result_df.iterrows(), 1):
        horse_num = str(row.get("horse_number", "-"))
        horse_name = str(row.get("horse_name", "-"))[:15]
        prob = row["prob_top3"]
        popularity = row.get("popularity", "-")
        odds = row.get("odds", "-")
        lines.append(
            f"{rank:>4}位  {horse_num:>3}番  {horse_name:<16}  {prob*100:>5.1f}%  "
            f"{str(popularity):>3}人気  {str(odds):>6}倍"
        )

    # ── 本命・対抗・穴馬 ───────────────────────────────────
    lines.append("")
    lines.append("■ 予想印")

    top = result_df.head(10)  # 上位10頭から役割を割り振る

    def _horse_label(row: pd.Series) -> str:
        return f"[{int(row.get('horse_number', 0))}番] {row.get('horse_name', '?')} ({row['prob_top3']*100:.1f}%)"

    # 1位 = 本命
    honmei = result_df.iloc[0]
    lines.append(f"  ◎ 本命: {_horse_label(honmei)}")

    # 2位 = 対抗
    if len(result_df) >= 2:
        taikou = result_df.iloc[1]
        lines.append(f"  ○ 対抗: {_horse_label(taikou)}")

    # 穴馬: 3位以降でオッズが10倍以上の馬の中で最も確率が高い馬
    ana_candidates = result_df.iloc[2:].copy()
    try:
        ana_candidates["_odds_num"] = pd.to_numeric(ana_candidates["odds"], errors="coerce")
        ana_df = ana_candidates[ana_candidates["_odds_num"] >= 10.0]
        if not ana_df.empty:
            ana = ana_df.iloc[0]
        else:
            ana = result_df.iloc[2] if len(result_df) >= 3 else None
    except Exception:
        ana = result_df.iloc[2] if len(result_df) >= 3 else None

    if ana is not None:
        lines.append(f"  △ 穴馬: {_horse_label(ana)}")

    # ── 馬連・ワイドの推奨 ─────────────────────────────────
    lines.append("")
    lines.append("■ 推奨買い目")

    top3 = result_df.head(3)
    if len(top3) >= 2:
        # 本命・対抗・穴の馬番を取得
        selected = result_df.head(3)["horse_number"].tolist()

        lines.append("  馬連 (上位3頭のボックス):")
        for a, b in combinations(selected, 2):
            lines.append(f"    {int(a)}-{int(b)}")

        lines.append("  ワイド (上位3頭のボックス):")
        for a, b in combinations(selected, 2):
            lines.append(f"    {int(a)}-{int(b)}")

        # 三連複
        if len(selected) >= 3:
            lines.append("  三連複:")
            lines.append(f"    {int(selected[0])}-{int(selected[1])}-{int(selected[2])}")

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
