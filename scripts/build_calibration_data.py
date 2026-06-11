"""
results_history.csv(本番予測の実績)からキャリブレータを生成する。

本番運用で記録された「予測確率 vs 実際の結果」は、未来データに対する
真の out-of-sample 予測なので、キャリブレーション学習に最適なデータ。
ただし上位3頭(本命・対抗・単穴)しか記録されていないため確率帯に
偏りがある。観測範囲外への外挿が安全な Platt scaling をデフォルトとする。

実行:
  python scripts/build_calibration_data.py            # Platt(推奨)
  python scripts/build_calibration_data.py --method auto

出力:
  keiba_predictor/model/calibrator.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from keiba_predictor.model.calibration import (  # noqa: E402
    Calibrator,
    expected_calibration_error,
)

HISTORY_PATH = ROOT / "keiba_predictor" / "data" / "results_history.csv"
OUTPUT_PATH = ROOT / "keiba_predictor" / "model" / "calibrator.pkl"
MIN_SAMPLES = 300  # これ未満なら学習しない(過学習防止)


def build_long_format(history: pd.DataFrame) -> pd.DataFrame:
    """results_history.csv(1行=1レース)を 1行=1頭 の縦持ちに変換する。"""
    rows = []
    for i in (1, 2, 3):
        sub = history[["date", "race_id", f"pred{i}_num", f"pred{i}_prob",
                       "actual1_num", "actual2_num", "actual3_num"]].copy()
        sub["prob"] = sub[f"pred{i}_prob"] / 100.0  # %表記 → 0-1
        sub["actual"] = sub.apply(
            lambda r: int(r[f"pred{i}_num"] in
                          (r["actual1_num"], r["actual2_num"], r["actual3_num"])),
            axis=1,
        )
        rows.append(sub[["date", "race_id", "prob", "actual"]])
    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=["prob", "actual"])
    df = df[(df["prob"] > 0) & (df["prob"] < 1)]
    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["platt", "isotonic", "auto"],
                        default="platt",
                        help="補正手法(上位3頭のみの偏ったデータでは platt 推奨)")
    parser.add_argument("--history", type=Path, default=HISTORY_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if not args.history.exists():
        sys.exit(f"履歴ファイルがありません: {args.history}")

    history = pd.read_csv(args.history)
    df = build_long_format(history)
    print(f"学習データ: {len(df)}頭分 ({df['date'].min()} 〜 {df['date'].max()})")

    if len(df) < MIN_SAMPLES:
        sys.exit(f"データ不足({len(df)} < {MIN_SAMPLES})。履歴が貯まるまで待つこと。")

    cal = Calibrator(method=args.method).fit(df["actual"].values, df["prob"].values)
    p_cal = cal.transform(df["prob"].values)

    print(f"\n採用手法: {cal.method}")
    print(f"観測確率範囲: {cal.prob_range[0]:.2f} 〜 {cal.prob_range[1]:.2f}")
    print(f"ECE(全データ): raw={expected_calibration_error(df['actual'].values, df['prob'].values):.4f}"
          f" → calibrated={expected_calibration_error(df['actual'].values, p_cal):.4f}")

    # 確率帯別の補正前後テーブル
    tbl = pd.DataFrame({"prob": df["prob"], "cal": p_cal, "actual": df["actual"]})
    tbl["bin"] = pd.cut(tbl["prob"], [0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0])
    summary = tbl.groupby("bin", observed=True).agg(
        n=("actual", "size"),
        生確率=("prob", "mean"),
        補正後=("cal", "mean"),
        実頻度=("actual", "mean"),
    ).round(3)
    print(f"\n{summary.to_string()}")

    with open(args.out, "wb") as f:
        pickle.dump(cal, f)
    print(f"\n保存: {args.out}")
    print("→ 次回の predict 実行から自動適用されます")


if __name__ == "__main__":
    main()
