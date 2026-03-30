"""
予測モジュール

- 学習済みモデルを使って指定レースの3着以内確率を出力
- 本命・対抗・穴馬の推奨
- 期待値スコア（EV）・危険な人気馬判定
- 馬連・ワイドの推奨組み合わせ
"""

import pickle
import io
import logging
import re
import sys


def _ensure_utf8_stdout() -> None:
    """Windows の cp932 端末で絵文字等が UnicodeEncodeError になるのを防ぐ。"""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


_ensure_utf8_stdout()
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from keiba_predictor.features.feature_engineering import FEATURE_COLS

try:
    import shap
except ImportError:
    shap = None

logger = logging.getLogger(__name__)

# SHAP値表示用の特徴量ラベルマッピング（日本語）
FEATURE_LABELS: dict[str, str] = {
    "distance": "距離適性",
    "course_type_enc": "コース適性",
    "track_condition_enc": "馬場状態",
    "weather_enc": "天候",
    "frame_number": "枠番",
    "horse_number": "馬番",
    "weight_carried": "斤量",
    "odds": "オッズ有利",
    "popularity": "人気",
    "sex_enc": "性別",
    "age": "年齢",
    "horse_weight": "馬体重",
    "horse_weight_diff": "馬体重増減",
    "last_3f": "上がり3F",
    "avg_time_3": "近3走タイム",
    "avg_time_5": "近5走タイム",
    "avg_time_3_any": "近3走タイム(全)",
    "avg_time_5_any": "近5走タイム(全)",
    "jockey_fukusho_rate": "騎手好成績",
    "trainer_fukusho_rate": "調教師好成績",
    "dist_diff_prev": "距離変更",
    "days_since_last_race": "レース間隔",
    "prev_finish_pos": "前走着順",
    "prev_odds": "前走オッズ",
    "horse_course_fukusho_rate": "コース実績",
    "horse_dist_fukusho_rate": "距離実績",
    "race_grade_enc": "レース格",
    "jockey_horse_fukusho_rate": "騎手馬相性",
    "horse_track_fukusho_rate": "馬場実績",
    "running_style_enc": "脚質",
    "pace_pressure": "展開圧力",
    "jockey_course_fukusho_rate": "騎手コース相性",
    "jockey_dist_fukusho_rate": "騎手距離適性",
    "weeks_since_last_race": "レース間隔",
    "is_fresh": "休み明け",
    "is_continuous": "連闘・中2週",
    "jockey_trainer_fukusho_rate": "騎手調教師相性",
    "weight_carried_diff": "斤量増減",
    "is_weight_increase": "斤量増加",
    "same_day_rank": "レース内順位",
    "prob_vs_avg": "平均比確率",
}

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "xgb_model.pkl"

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


def compute_shap_top(
    model_bundle: dict,
    X: pd.DataFrame,
    feature_cols: list[str],
) -> list[list[dict]]:
    """
    各馬のSHAP値を計算し、上位3特徴量（プラス最大2 + マイナス最大1）を返す。

    Returns:
        馬ごとの shap_top リスト。各要素は
        [{"feature": str, "value": float, "label": str}, ...] の形式。
    """
    if shap is None:
        logger.warning("shapパッケージ未インストール: SHAP値計算をスキップ")
        return [[] for _ in range(len(X))]

    try:
        model = model_bundle["model"]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception as e:
        logger.warning(f"SHAP値計算に失敗: {e}")
        return [[] for _ in range(len(X))]

    results = []
    for i in range(len(X)):
        sv = shap_values[i]
        pairs = list(zip(feature_cols, sv))

        # プラス方向の上位2つ
        positive = sorted([p for p in pairs if p[1] > 0], key=lambda x: -x[1])[:2]
        # マイナス方向の上位1つ
        negative = sorted([p for p in pairs if p[1] < 0], key=lambda x: x[1])[:1]

        top = []
        for feat, val in positive:
            label = FEATURE_LABELS.get(feat, feat)
            top.append({"feature": feat, "value": round(float(val), 4), "label": label})
        for feat, val in negative:
            label = FEATURE_LABELS.get(feat, feat)
            # マイナス要因のラベルに「やや悪」等のニュアンスを付加
            if not any(neg in label for neg in ["悪", "不", "低"]):
                label = label + "やや悪"
            top.append({"feature": feat, "value": round(float(val), 4), "label": label})

        results.append(top)
    return results


def load_band_model(distance: float) -> Optional[dict]:
    """距離から距離帯別モデルをロードする。存在しなければNoneを返す。"""
    from keiba_predictor.model.train import classify_distance_band, DISTANCE_BAND_LABELS
    band = classify_distance_band(distance)
    band_path = MODEL_DIR / f"xgb_model_{band}.pkl"
    if not band_path.exists():
        logger.info(f"距離帯モデル ({DISTANCE_BAND_LABELS[band]}) が見つかりません → 統合モデルを使用")
        return None
    with open(band_path, "rb") as f:
        bundle = pickle.load(f)
    label = DISTANCE_BAND_LABELS[band]
    logger.info(f"距離帯モデル使用: {label} ({int(distance)}m) AUC: {bundle.get('cv_auc_mean', 'N/A'):.4f}")
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

    # 同レース内相対評価
    result["same_day_rank"] = result["prob_top3"].rank(ascending=False, method="first")
    avg_prob = result["prob_top3"].mean()
    result["prob_vs_avg"] = result["prob_top3"] - avg_prob

    # SHAP値を計算して各馬に付与
    shap_tops = compute_shap_top(model_bundle, X, feature_cols)
    result["shap_top"] = shap_tops

    result = result.sort_values("prob_top3", ascending=False).reset_index(drop=True)
    return result


def _decide_bet_strategy(result_df: pd.DataFrame) -> dict:
    """
    予測結果DataFrameから最適な買い目を自動決定する。

    判断ロジック:
    【複勝】本命EV>1.0 & 1番人気でない → 買い
    【馬連/ワイド】上位3頭の確率差5%以内 → ワイド、それ以外 → 馬連
    【3連複】本命確率50%以上→軸1×相手4(6点)、40-50%→軸1×相手5(10点)、拮抗→2頭軸×3(3点)
    【穴馬】穴馬EV≥10 → 穴馬軸の馬連1点追加
    """
    from itertools import combinations as _comb

    if len(result_df) < 3:
        return {
            "fukusho": [], "umaren": [], "wide": [],
            "sanrenpuku": {}, "total_points": 0,
            "strategy_note": "出走頭数不足", "use_wide": False,
        }

    top5 = result_df.head(5)
    nums = [int(r["horse_number"]) for _, r in top5.iterrows()
            if pd.notna(r.get("horse_number"))]
    names = {}
    for _, r in result_df.iterrows():
        n = r.get("horse_number")
        if pd.notna(n):
            names[int(n)] = str(r.get("horse_name", ""))

    hon = nums[0]
    hon_prob = float(result_df.iloc[0]["prob_top3"])
    hon_ev = float(result_df.iloc[0].get("ev_score", 0)) if pd.notna(result_df.iloc[0].get("ev_score")) else 0
    hon_pop = pd.to_numeric(result_df.iloc[0].get("popularity"), errors="coerce")

    probs = [float(result_df.iloc[i]["prob_top3"]) for i in range(min(3, len(result_df)))]
    prob_spread = max(probs) - min(probs) if len(probs) >= 3 else 1.0
    is_tight = prob_spread < 0.05  # 5%以内 = 拮抗

    # 穴馬
    ana_num = None
    ana_ev = 0
    top5_set = set(nums)
    if len(result_df) > 5:
        rest = result_df.iloc[5:]
        rest_prob = pd.to_numeric(rest.get("prob_top3", pd.Series(dtype=float)), errors="coerce")
        rest_pop = pd.to_numeric(rest.get("popularity", pd.Series(dtype=float)), errors="coerce")
        cands = rest[(rest_prob >= 0.35) & (rest_pop >= 6)]
        if not cands.empty:
            best = cands.nlargest(1, "prob_top3").iloc[0]
            v = best.get("horse_number")
            if pd.notna(v) and int(v) not in top5_set:
                ana_num = int(v)
                ana_ev = float(best.get("ev_score", 0)) if pd.notna(best.get("ev_score")) else 0

    strategy = {
        "fukusho": [],
        "umaren": [],
        "wide": [],
        "sanrenpuku": {},
        "total_points": 0,
        "strategy_note": "",
        "use_wide": is_tight,
    }
    notes = []

    # 【複勝】
    if hon_ev > 1.0 and (pd.isna(hon_pop) or hon_pop > 1):
        strategy["fukusho"] = [{"num": hon, "name": names.get(hon, "")}]
        notes.append("複勝")

    # 【馬連 or ワイド】
    use_wide = is_tight
    if use_wide:
        pairs = [{"nums": list(p)} for p in _comb(nums[:3], 2)]
        strategy["wide"] = pairs
        notes.append("ワイド（上位拮抗）")
    else:
        pairs = [{"nums": list(p)} for p in _comb(nums[:3], 2)]
        strategy["umaren"] = pairs
        notes.append("馬連")

    # 【穴馬絡み馬連】
    if ana_num and ana_ev >= 10:
        strategy["umaren"].append({"nums": [hon, ana_num]})
        notes.append("穴馬馬連追加")

    # 【3連複】
    if hon_prob >= 0.50:
        # 軸1頭×相手4頭 = C(4,2) = 6点
        aite = nums[1:5]
        strategy["sanrenpuku"] = {"jiku": [hon], "aite": aite}
        notes.append("3連複軸1×4")
    elif hon_prob >= 0.40:
        # 軸1頭×相手5頭 = C(5,2) = 10点
        aite = nums[1:5]
        if ana_num and ana_num not in aite:
            aite = aite + [ana_num]
        strategy["sanrenpuku"] = {"jiku": [hon], "aite": aite}
        notes.append("3連複軸1×5")
    elif is_tight and len(nums) >= 5:
        # 2頭軸×3頭 = C(3,1)*1 = 3点
        strategy["sanrenpuku"] = {"jiku": nums[:2], "aite": nums[2:5]}
        notes.append("3連複2頭軸×3")

    # 合計点数
    total = len(strategy["fukusho"])
    total += len(strategy["umaren"])
    total += len(strategy["wide"])
    sr = strategy["sanrenpuku"]
    if sr:
        jiku = sr.get("jiku", [])
        aite = sr.get("aite", [])
        if len(jiku) == 1:
            total += len(list(_comb(aite, 2)))
        elif len(jiku) == 2:
            total += len(aite)
    strategy["total_points"] = total
    strategy["strategy_note"] = " + ".join(notes) if notes else "見送り"

    return strategy


def _build_buy_lines(result_df: pd.DataFrame, race_name: str = "") -> list[str]:
    """
    買い目リストを返す。
    3連複: 軸（◎）× 相手（2〜5位 + 穴馬）
    穴馬=AI確率35%以上&6番人気以下&TOP5外。穴馬あり→10点、穴馬なし→6点。
    """
    from itertools import combinations as _comb

    SEP = "━" * 20

    # 上位5頭の馬番
    top5 = result_df.head(5)
    nums = [int(r["horse_number"]) for _, r in top5.iterrows()
            if pd.notna(r.get("horse_number"))]

    if len(nums) < 2:
        return []

    hon = nums[0]  # 軸（◎）

    # 複勝: 本命馬名
    hon_name = ""
    hon_row = result_df.iloc[0]
    if pd.notna(hon_row.get("horse_name")):
        hon_name = str(hon_row["horse_name"])

    # 馬連: top3の組み合わせ
    umaren_pairs = list(_comb(nums[:3], 2))
    umaren_str   = " / ".join(f"{a}-{b}" for a, b in umaren_pairs)

    # 穴馬: AI確率35%以上 & 6番人気以下 & TOP5外 → AI確率最高の1頭
    ana_num = None
    top5_set = set(nums)
    if len(result_df) > 5:
        rest = result_df.iloc[5:]
        rest_prob = pd.to_numeric(rest.get("prob_top3", pd.Series(dtype=float)), errors="coerce")
        rest_pop = pd.to_numeric(rest.get("popularity", pd.Series(dtype=float)), errors="coerce")
        cands = rest[(rest_prob >= 0.35) & (rest_pop >= 6)]
        if not cands.empty:
            best = cands.nlargest(1, "prob_top3").iloc[0]
            v = best.get("horse_number")
            if pd.notna(v) and int(v) not in top5_set:
                ana_num = int(v)

    # 3連複: 軸1頭 × 相手（2〜5位 + 穴馬）
    partners = nums[1:5]  # 2〜5位
    if ana_num and ana_num not in partners:
        partners = partners + [ana_num]
    sanren_pt    = len(list(_comb(partners, 2)))  # C(n,2)
    partners_str = "/".join(
        f"{n}（穴）" if n == ana_num else str(n) for n in partners
    )

    total = 1 + len(umaren_pairs) + sanren_pt

    header = f"💰 {race_name}  買い目" if race_name else "💰 買い目"
    lines = [
        SEP,
        header,
        SEP,
        "■ 複勝（1点）",
        f"　{hon}番 {hon_name}",
        f"■ 馬連（{len(umaren_pairs)}点）",
        f"　{umaren_str}",
        f"■ 3連複（{sanren_pt}点）",
        f"　軸 {hon}番",
        f"　× {partners_str}",
        SEP,
        f"合計 {total}点",
        SEP,
    ]
    return lines


def format_prediction(
    result_df: pd.DataFrame,
    race_name: str = "",
    ai_comments: Optional[dict] = None,
    course_info: str = "",
) -> tuple[str, str]:
    """
    予測結果を競馬新聞風の2メッセージ（予想・買い目）で返す。

    Returns:
        (msg1_予想, msg2_買い目) のタプル
    """
    if "ev_score" not in result_df.columns:
        result_df = calc_ev_and_flags(result_df)

    if ai_comments is None:
        ai_comments = {}

    sep = "━" * 20

    # ── Message 1: 予想（1馬1行・コンパクト） ─────────────────
    race_label = race_name if race_name else "KEIBA EDGE 予測結果"
    lines1 = [sep, f"🏇 {race_label}"]
    if course_info:
        lines1.append(course_info)
    lines1.append(sep)

    MARKS = ["◎", "○", "▲", "△", "　"]
    top5  = result_df.head(5)

    for rank, (_, row) in enumerate(top5.iterrows()):
        mark     = MARKS[rank] if rank < len(MARKS) else "　"
        num      = str(int(row["horse_number"])) if pd.notna(row.get("horse_number")) else "-"
        name     = str(row.get("horse_name", "-"))
        prob     = row["prob_top3"] * 100
        ev       = row.get("ev_score")
        ev_str   = f" EV{ev:.2f}" if pd.notna(ev) else ""
        lines1.append(f"{mark} {num}番 {name}　{prob:.1f}%{ev_str}")

    lines1.append(sep)

    # ★穴馬（TOP5外・AI確率35%以上・6番人気以下）
    top5_idx = top5.index
    ana_df = result_df.loc[
        ~result_df.index.isin(top5_idx) &
        (result_df["prob_top3"] >= 0.35) &
        (pd.to_numeric(result_df.get("popularity", pd.Series(dtype=float)), errors="coerce") >= 6)
    ]
    if not ana_df.empty:
        row      = ana_df.nlargest(1, "prob_top3").iloc[0]
        num      = int(row["horse_number"]) if pd.notna(row.get("horse_number")) else 0
        name     = str(row.get("horse_name", ""))
        prob     = row["prob_top3"] * 100
        pop      = str(int(row["popularity"])) if pd.notna(row.get("popularity")) else "-"
        lines1.append(f"★穴 {num}番{name}（AI確率{prob:.1f}% {pop}番人気）")
        lines1.append(f"　→ AIが高評価も市場は低評価！")

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
    msg2 = "\n".join(_build_buy_lines(result_df, race_name=race_name))

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

    df = pd.read_csv(featured_path, encoding="utf-8-sig")
    if "race_date" in df.columns:
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    race_df = df[df["race_id"].astype(str) == str(race_id)].copy()

    if race_df.empty:
        raise ValueError(f"race_id={race_id} がデータに存在しません")

    # 距離帯別モデルを優先使用
    band_bundle = None
    if "distance" in race_df.columns and model_path is None:
        dist = pd.to_numeric(race_df["distance"].iloc[0], errors="coerce")
        if pd.notna(dist):
            try:
                band_bundle = load_band_model(float(dist))
            except Exception as e:
                logger.warning(f"距離帯モデルロード失敗: {e}")
    model_bundle = band_bundle if band_bundle else load_model(model_path)
    result = predict_race(race_df, model_bundle)
    result = calc_ev_and_flags(result)

    race_name   = race_df["race_name"].iloc[0] if "race_name" in race_df.columns else race_id
    course_info = _build_course_info(race_id, race_df)

    from keiba_predictor.ai_comment import generate_comments, generate_report_text, save_report
    try:
        ai_comments = generate_comments(result, race_name=race_name, course_info=course_info)
    except Exception as e:
        logger.warning(f"AI解説生成でエラー（続行）: {e}")
        ai_comments = {}

    msg1, msg2 = format_prediction(result, race_name=race_name, ai_comments=ai_comments,
                                   course_info=course_info)
    print(msg1)
    print(msg2)

    # note / BOOKERS 投稿用レポートをファイルに保存（Discord通知とは独立）
    if ai_comments:
        report = generate_report_text(ai_comments, race_name=race_name,
                                      course_info=course_info, result_df=result,
                                      buy_lines=_build_buy_lines(result, race_name=race_name))
        save_report(report, race_name)

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

    from keiba_predictor.ai_comment import flush_reports
    flush_reports()

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

    # 予測（距離帯別モデルを優先使用）
    band_bundle = None
    distance = shutuba_info.get("distance")
    if distance is not None and model_path is None:
        try:
            band_bundle = load_band_model(float(distance))
        except Exception as e:
            logger.warning(f"距離帯モデルロード失敗: {e}")
    model_bundle = band_bundle if band_bundle else load_model(model_path)
    result = predict_race(race_df, model_bundle)
    result = calc_ev_and_flags(result)

    race_name   = shutuba_info.get("race_name", "")
    course_info = shutuba_info.get("course_info", "")

    from keiba_predictor.ai_comment import generate_comments, generate_report_text, save_report
    try:
        ai_comments = generate_comments(result, race_name=race_name, course_info=course_info)
    except Exception as e:
        logger.warning(f"AI解説生成でエラー（続行）: {e}")
        ai_comments = {}

    msg1, msg2 = format_prediction(result, race_name=race_name, ai_comments=ai_comments, course_info=course_info)
    print(msg1)
    print(msg2)

    # note / BOOKERS 投稿用レポートをファイルに保存（Discord通知とは独立）
    if ai_comments:
        report = generate_report_text(ai_comments, race_name=race_name,
                                      course_info=course_info, result_df=result,
                                      buy_lines=_build_buy_lines(result, race_name=race_name))
        save_report(report, race_name)

    # 予想キャッシュに保存（note_report・結果照合で使用）
    race_date  = shutuba_info.get("race_date", "")
    start_time = shutuba_info.get("start_time", "")
    venue      = shutuba_info.get("venue", "")
    from keiba_predictor.discord_notify import _store_prediction
    _store_prediction(race_id, race_name, race_date, result,
                      ai_comments=ai_comments, course_info=course_info,
                      start_time=start_time, venue=venue)

    if notify:
        import os
        from keiba_predictor.discord_notify import send_discord
        url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")
        if not url:
            logger.error("--webhook-url または環境変数 DISCORD_WEBHOOK_URL を指定してください")
        else:
            ok = send_discord(url, msg1) and send_discord(url, msg2)
            logger.info(f"Discord 送信{'完了' if ok else '失敗'}")

    from keiba_predictor.ai_comment import flush_reports
    flush_reports()

    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m keiba_predictor.model.predict <race_id>")
        sys.exit(1)
    predict_from_csv(sys.argv[1])
