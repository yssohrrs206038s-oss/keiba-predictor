"""
確率キャリブレーションモジュール

モデルの生出力確率を実際の的中頻度に補正する後処理。
モデル本体(xgb_model.pkl)には一切手を加えない。

背景:
  results_history.csv の検証で、予測96%台の馬の実際の3着以内率が
  53%程度であることが判明(深刻な過信)。EVスコア・自信度判定が
  このズレの影響を受けるため、確率を実頻度に合わせて補正する。

使い方:
  学習:  python scripts/build_calibration_data.py
  適用:  predict.py 内で apply_calibration() が自動適用
        (calibrator.pkl が無ければ生確率のまま動作)
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_warned_missing = False  # calibrator未配置の警告を1回に抑制


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE: 予測確率と実頻度のズレの加重平均。小さいほど良い。"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return ece


class Calibrator:
    """
    Isotonic / Platt を時系列内部検証で比較し、検証ECEが小さい方を採用。

    注意: results_history.csv 由来のデータは上位3頭のみのため確率帯に
    偏りがある。観測範囲外への外挿は Platt(滑らかな単調変換)の方が
    安全なので、--method platt の明示指定を推奨。
    """

    def __init__(self, method: str = "auto"):
        self.requested_method = method
        self.method: Optional[str] = None
        self.iso = None
        self.platt = None
        self.prob_range: tuple = (0.0, 1.0)  # 学習データの観測範囲
        self.report: dict = {}

    def fit(self, y_true, y_prob) -> "Calibrator":
        """y_true, y_prob は時系列順(古い→新しい)で渡すこと。"""
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        self.prob_range = (float(y_prob.min()), float(y_prob.max()))

        # 内部検証: 末尾20%(=直近データ)で手法選択
        n_val = max(int(len(y_true) * 0.2), 30)
        tr_y, tr_p = y_true[:-n_val], y_prob[:-n_val]
        va_y, va_p = y_true[-n_val:], y_prob[-n_val:]

        def fit_iso(y, p):
            m = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
            m.fit(p, y)
            return m

        def fit_platt(y, p):
            m = LogisticRegression(C=1e6)
            m.fit(_logit(p).reshape(-1, 1), y)
            return m

        if self.requested_method == "auto":
            iso_va = fit_iso(tr_y, tr_p).predict(va_p)
            platt_m = fit_platt(tr_y, tr_p)
            platt_va = platt_m.predict_proba(_logit(va_p).reshape(-1, 1))[:, 1]
            ece_iso = expected_calibration_error(va_y, iso_va)
            ece_platt = expected_calibration_error(va_y, platt_va)
            self.method = "isotonic" if ece_iso < ece_platt else "platt"
            self.report["ece_val_isotonic"] = ece_iso
            self.report["ece_val_platt"] = ece_platt
        else:
            self.method = self.requested_method

        # 採用手法を全データで再学習
        if self.method == "isotonic":
            self.iso = fit_iso(y_true, y_prob)
        else:
            self.platt = fit_platt(y_true, y_prob)

        self.report["method"] = self.method
        self.report["n_samples"] = len(y_true)
        self.report["ece_raw_val"] = expected_calibration_error(va_y, va_p)
        return self

    def transform(self, y_prob) -> np.ndarray:
        y_prob = np.asarray(y_prob, dtype=float)
        if self.method == "isotonic":
            return self.iso.predict(y_prob)
        return self.platt.predict_proba(_logit(y_prob).reshape(-1, 1))[:, 1]


def apply_calibration(probs: np.ndarray, calibrator_path: Path) -> np.ndarray:
    """
    predict処理から呼ぶ。calibrator.pkl が無ければ生確率をそのまま返す
    (既存動作を壊さないためのフォールバック)。
    """
    global _warned_missing
    path = Path(calibrator_path)
    if not path.exists():
        if not _warned_missing:
            logger.warning(
                f"キャリブレータ未配置のため生確率を使用: {path} "
                "(scripts/build_calibration_data.py で生成可能)"
            )
            _warned_missing = True
        return probs
    with open(path, "rb") as f:
        cal: Calibrator = pickle.load(f)
    calibrated = cal.transform(probs)
    logger.debug(f"確率キャリブレーション適用 ({cal.method})")
    return calibrated
