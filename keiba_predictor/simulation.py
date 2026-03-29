"""
モンテカルロシミュレーションによる着順分布・軸馬判定

各馬の確率・脚質・展開をランダムに変動させて
1万回レースをシミュレーションし着順分布を算出する。
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# 展開パターン確率
PACE_PATTERNS = {
    "high_pace": 0.30,   # ハイペース
    "slow_pace": 0.40,   # スローペース
    "normal":    0.30,    # 標準
}

# 脚質別の展開補正 (running_style_enc: 0=逃げ, 1=先行, 2=差し, 3=追い込み)
PACE_ADJUSTMENTS = {
    "high_pace": {0: -0.10, 1: -0.10, 2: 0.10, 3: 0.10},
    "slow_pace": {0: 0.10, 1: 0.10, 2: -0.10, 3: -0.10},
    "normal":    {0: 0.0,  1: 0.0,  2: 0.0,  3: 0.0},
}

STUMBLE_PROB = 0.05      # 出遅れ確率
STUMBLE_PENALTY = -0.15  # 出遅れ時の確率補正

STABILITY_THRESHOLD = 0.10  # 標準偏差がこの値未満なら「安定軸」


def run_monte_carlo(
    horses: list[dict],
    n_simulations: int = 10000,
    seed: Optional[int] = 42,
) -> dict:
    """
    各馬の確率・脚質・展開をランダムに変動させて
    n_simulations回レースをシミュレーションし着順分布を算出。

    Args:
        horses: 馬情報のリスト。各要素は以下のキーを含む:
            - horse_number: int  馬番
            - horse_name: str    馬名
            - prob: float        AI 3着以内確率 (0-1)
            - running_style_enc: int  脚質 (0=逃げ,1=先行,2=差し,3=追い込み)
        n_simulations: シミュレーション回数
        seed: 乱数シード（再現性のため）

    Returns:
        馬番(str) → シミュレーション結果 の辞書
    """
    if not horses:
        return {}

    rng = np.random.default_rng(seed)
    n_horses = len(horses)

    # 展開パターンの事前サンプリング
    pace_names = list(PACE_PATTERNS.keys())
    pace_probs = [PACE_PATTERNS[p] for p in pace_names]
    pace_indices = rng.choice(len(pace_names), size=n_simulations, p=pace_probs)

    # 各馬の基本情報を配列化
    base_probs = np.array([h["prob"] for h in horses], dtype=np.float64)
    styles = [int(h.get("running_style_enc", 2)) for h in horses]

    # 展開パターンごとの補正テーブル
    adj_table = np.zeros((len(pace_names), n_horses), dtype=np.float64)
    for pi, pace in enumerate(pace_names):
        for hi, style in enumerate(styles):
            adj_table[pi, hi] = PACE_ADJUSTMENTS[pace].get(style, 0.0)

    # シミュレーション本体
    # shape: (n_simulations, n_horses)
    sim_probs = np.tile(base_probs, (n_simulations, 1))

    # 展開補正を適用
    sim_probs += adj_table[pace_indices]

    # 出遅れシミュレーション
    stumble_mask = rng.random((n_simulations, n_horses)) < STUMBLE_PROB
    sim_probs += stumble_mask * STUMBLE_PENALTY

    # ランダムノイズ（各馬±5%程度のゆらぎ）
    noise = rng.normal(0, 0.05, size=(n_simulations, n_horses))
    sim_probs += noise

    # 確率を0-1にクリップ
    sim_probs = np.clip(sim_probs, 0.01, 0.99)

    # 各シミュレーションで着順を決定（確率の高い順 → 上位3着）
    # 確率にノイズを加えた値でソートし、上位3頭を「3着以内」とする
    performance = sim_probs + rng.normal(0, 0.1, size=(n_simulations, n_horses))
    rankings = np.argsort(-performance, axis=1)  # 降順
    top3_mask = np.zeros((n_simulations, n_horses), dtype=bool)
    for k in range(min(3, n_horses)):
        top3_mask[np.arange(n_simulations), rankings[:, k]] = True

    # 展開パターン別に集計
    scenario_top3 = {pace: np.zeros(n_horses, dtype=np.float64) for pace in pace_names}
    scenario_count = {pace: 0 for pace in pace_names}

    for pi, pace in enumerate(pace_names):
        mask = pace_indices == pi
        count = mask.sum()
        if count > 0:
            scenario_top3[pace] = top3_mask[mask].mean(axis=0)
            scenario_count[pace] = int(count)

    # 全体の3着以内率と標準偏差
    overall_top3_rate = top3_mask.mean(axis=0)

    # ブロック単位での標準偏差（安定度の指標）
    block_size = max(100, n_simulations // 20)
    n_blocks = n_simulations // block_size
    if n_blocks >= 2:
        block_rates = np.zeros((n_blocks, n_horses))
        for b in range(n_blocks):
            s = b * block_size
            e = s + block_size
            block_rates[b] = top3_mask[s:e].mean(axis=0)
        top3_std = block_rates.std(axis=0)
    else:
        top3_std = np.zeros(n_horses)

    # 結果を辞書に変換
    results = {}
    for i, h in enumerate(horses):
        num_str = str(h["horse_number"])
        is_stable = float(top3_std[i]) < STABILITY_THRESHOLD
        results[num_str] = {
            "horse_name": h.get("horse_name", ""),
            "top3_rate": round(float(overall_top3_rate[i]), 3),
            "top3_std": round(float(top3_std[i]), 3),
            "is_stable": is_stable,
            "scenario": {
                pace: round(float(scenario_top3[pace][i]), 2)
                for pace in pace_names
            },
        }

    logger.info(
        f"モンテカルロシミュレーション完了: {n_simulations}回, {n_horses}頭"
    )

    return results
