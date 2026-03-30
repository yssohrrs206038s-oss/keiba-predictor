"""
大口投票検知モジュール

オッズ断層・単勝/複勝乖離から異常度スコアを算出し、
大口投票の可能性がある馬を検知する。
"""

import logging

logger = logging.getLogger(__name__)


def detect_large_bets(race_id: str, horses: list[dict]) -> list[dict]:
    """
    各馬の異常度スコアを算出し、ランクA/Bの馬を返す。

    Args:
        race_id: レースID
        horses: [{"horse_number": int, "horse_name": str,
                  "odds": float, "popularity": int,
                  "fukusho_odds": float or None}, ...]

    Returns:
        ランクA/Bの馬のリスト。各要素は:
        {"horse_number", "horse_name", "rank", "score",
         "gap_score", "divergence_score", "odds", "fukusho_odds",
         "gap_detail", "divergence_detail"}
    """
    if len(horses) < 3:
        return []

    # オッズが有効な馬のみ処理
    valid = [h for h in horses if h.get("odds") and float(h["odds"]) > 0]
    if len(valid) < 3:
        return []

    # 人気順（オッズ昇順）にソート
    sorted_h = sorted(valid, key=lambda x: float(x["odds"]))

    # ── 指標1: オッズ断層 ────────────────────────────────────
    gap_scores: dict[int, tuple[int, str]] = {}  # horse_number → (score, detail)
    for i in range(1, len(sorted_h)):
        prev_odds = float(sorted_h[i - 1]["odds"])
        curr_odds = float(sorted_h[i]["odds"])
        if prev_odds <= 0:
            continue
        ratio = curr_odds / prev_odds
        if ratio >= 1.5:
            num = sorted_h[i]["horse_number"]
            gap_scores[num] = (40, f"前馬との倍率差{ratio:.1f}倍")

    # ── 指標2: 単勝/複勝乖離 ────────────────────────────────
    div_scores: dict[int, tuple[int, str]] = {}  # horse_number → (score, detail)
    for h in valid:
        odds = float(h["odds"])
        fukusho = h.get("fukusho_odds")
        if not fukusho or float(fukusho) <= 0 or odds <= 0:
            continue
        fukusho = float(fukusho)

        # 推定得票率
        tansho_rate = 1.0 / odds
        fukusho_rate = 1.0 / fukusho / 3.0
        if tansho_rate <= 0:
            continue
        divergence = fukusho_rate / tansho_rate

        if divergence > 1.5:
            num = h["horse_number"]
            score = min(60, int(divergence * 30))
            div_scores[num] = (
                score,
                f"単勝{odds:.1f}倍 / 複勝{fukusho:.1f}倍（複勝に大量資金の可能性）",
            )

    # ── 合算＆ランク判定 ──────────────────────────────────────
    all_nums = set(gap_scores.keys()) | set(div_scores.keys())
    results = []

    for h in valid:
        num = h["horse_number"]
        if num not in all_nums:
            continue

        g_score, g_detail = gap_scores.get(num, (0, ""))
        d_score, d_detail = div_scores.get(num, (0, ""))
        total = min(100, g_score + d_score)

        if total < 50:
            continue

        rank = "A" if total >= 80 else "B"
        results.append({
            "horse_number": num,
            "horse_name": h.get("horse_name", ""),
            "rank": rank,
            "score": total,
            "gap_score": g_score,
            "divergence_score": d_score,
            "odds": float(h["odds"]),
            "fukusho_odds": float(h["fukusho_odds"]) if h.get("fukusho_odds") else None,
            "gap_detail": g_detail,
            "divergence_detail": d_detail,
        })

    results.sort(key=lambda x: -x["score"])
    return results


def format_large_bet_alert(race_name: str, flags: list[dict]) -> str:
    """大口投票検知結果をDiscord通知用テキストに変換する。"""
    if not flags:
        return ""

    SEP = "━" * 20
    lines = [
        f"🚨 【大口投票検知】{race_name}",
        SEP,
    ]

    for f in flags:
        num = f["horse_number"]
        name = f["horse_name"]
        rank = f["rank"]
        score = f["score"]
        odds = f["odds"]

        lines.append(f"{num}番 {name}")
        lines.append(f"判定：ランク{rank}（異常スコア {score}）")

        if f.get("divergence_detail"):
            lines.append(f"{f['divergence_detail']}")
        if f.get("gap_detail"):
            lines.append(f"オッズ断層：{f['gap_detail']}")
        lines.append("")

    lines.append(SEP)
    lines.append("⚠️ 偽陽性の可能性あり。参考情報としてご利用ください。")
    return "\n".join(lines)
