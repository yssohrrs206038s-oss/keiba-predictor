"""
note記事用 アナリストプロンプトによるレース分析レポート生成

predictions_cache.json のレースデータを元に
Claude API で本格的な競馬分析レポート（Markdown）を生成する。
"""

import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_ID = "claude-haiku-4-5-20251001"
DATA_DIR = Path(__file__).parent / "data"

SYSTEM_PROMPT = """\
あなたは「データ・展開・馬場・騎手・オッズ」を網羅した
トップ競馬アナリストです。
長期回収率を最大化するための「期待値重視」の分析を行います。
以下のルールを厳守してください：
- 「絶対当たる」「確実」などの断定表現は禁止
- 情報が足りない部分は「不明」と明記
- 人気馬を買う理由だけでなく「消す理由」も必ず示す
- 調教・血統データは提供されないため言及しない
- 提供データの数値のみを使うこと
- オッズ・配当金額はデータに含まれていないため書かないこと
"""

OUTPUT_TEMPLATE = """\
以下のレースデータを分析し、Markdown形式で記事を生成してください。

## 出力フォーマット（必ずこの構成で出力）

# 【{race_name}】AI予想分析

## レース概要
（日付・競馬場・距離・コース条件を簡潔にまとめる）

## 展開予測
（脚質分布からペース想定・有利な位置取りを分析）

## 危険な人気馬
（人気だがAI確率が低い馬、前走不振の馬を指摘。消す理由を具体的に）

---
※ここから有料（¥300）

## 各馬評価
（全出走馬のうち上位5頭の詳細分析。強み・弱みを具体的に）

## 印と理由
（◎○▲△☆の印を付けて各100字程度で理由を記述）

## 馬券戦略
### A. 的中率重視プラン
（複勝・ワイド中心。堅い組み合わせ）

### B. バランス型プラン
（馬連・ワイド中心。的中率と配当のバランス）

### C. 回収率重視プラン
（3連複・穴馬絡み。高配当狙い）

## 買い目
（具体的な馬番の組み合わせと点数。各プラン別に記載）

## 最終判断
（このレースで最も重要なポイントを1文で）

## 不確実性
（予想の限界・当日変動リスクを正直に記述）
"""


def _build_race_data_prompt(entry: dict) -> str:
    """predictions_cache.json のエントリからプロンプト用データ文字列を構築する。"""
    race_name = entry.get("race_name", "不明")
    race_date = entry.get("race_date", "不明")
    venue = entry.get("venue", "不明")
    course_info = entry.get("course_info", "不明")

    lines = [
        f"レース名: {race_name}",
        f"日付: {race_date}",
        f"競馬場: {venue}",
        f"コース: {course_info}",
        "",
        "## 出走馬データ",
    ]

    # predicted_top5 から詳細データ
    top5 = entry.get("predicted_top5", [])
    roles = {
        entry.get("honmei", {}).get("horse_number"): "◎",
        entry.get("taikou", {}).get("horse_number"): "○",
        entry.get("ana", {}).get("horse_number"): "▲",
    }

    ev_map = {}
    for e in entry.get("ev_top3", []):
        if e.get("horse_number"):
            ev_map[e["horse_number"]] = e

    for h in top5:
        num = h.get("horse_number", "?")
        name = h.get("horse_name", "?")
        prob = h.get("prob", 0) * 100
        odds = h.get("odds", "不明")
        mark = roles.get(num, "")

        ev_entry = ev_map.get(num, {})
        ev = ev_entry.get("ev_score", 0)

        line = f"- {mark}{num}番 {name}: AI確率{prob:.1f}%"
        if odds and odds != "不明":
            line += f" オッズ{odds}倍"
        if ev:
            line += f" EV{ev:.2f}"
        lines.append(line)

    # honmei/taikou/ana の追加データ（SHAP等）
    for role_key, mark in [("honmei", "◎"), ("taikou", "○"), ("ana", "▲")]:
        p = entry.get(role_key, {})
        if not p or not p.get("horse_name"):
            continue
        shap = p.get("shap_top", [])
        if shap:
            strengths = [s["label"] for s in shap if s.get("value", 0) > 0]
            concerns = [s["label"] for s in shap if s.get("value", 0) < 0]
            num = p.get("horse_number", "?")
            if strengths or concerns:
                detail = f"  {mark}{num}番 SHAP:"
                if strengths:
                    detail += f" 強み={','.join(strengths)}"
                if concerns:
                    detail += f" 弱み={','.join(concerns)}"
                lines.append(detail)
        # 前走データ
        for key, label in [("prev_passing", "前走通過"), ("prev_last_3f", "前走上がり3F")]:
            val = p.get(key)
            if val:
                lines.append(f"  {mark}{p.get('horse_number','?')}番 {label}: {val}")

    # 危険馬
    dangerous = entry.get("dangerous_horses", [])
    if dangerous:
        lines.append("")
        lines.append("## 危険馬指定")
        for d in dangerous:
            reasons = d.get("reasons", [])
            lines.append(f"- {d.get('horse_number','?')}番 {d.get('horse_name','?')}"
                         f"（{d.get('popularity','?')}人気）: {'; '.join(reasons)}")

    # 自信度
    conf = entry.get("confidence_stars", "")
    if conf:
        lines.append(f"\nAI自信度: {conf}")

    return "\n".join(lines)


def generate_analyst_report(race_id: str, cache: Optional[dict] = None) -> str:
    """
    1レース分のアナリストレポート（Markdown）を生成する。

    Returns:
        Markdown文字列。失敗時は空文字列。
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY 未設定 → note記事生成スキップ")
        return ""

    if cache is None:
        cache_path = DATA_DIR / "predictions_cache.json"
        if not cache_path.exists():
            return ""
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    entry = cache.get(race_id, {})
    if not entry or not entry.get("predicted_top3_nums"):
        logger.warning(f"レースデータなし: {race_id}")
        return ""

    race_name = entry.get("race_name", race_id)
    race_data = _build_race_data_prompt(entry)

    prompt = OUTPUT_TEMPLATE.format(race_name=race_name) + "\n\n## 入力データ\n\n" + race_data

    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic 未インストール")
        return ""

    client = anthropic.Anthropic(api_key=api_key)

    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(10 * attempt)
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=3000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            report = response.content[0].text.strip()
            logger.info(f"note記事生成完了: {race_name} ({len(report)}字)")
            return report
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str:
                logger.warning(f"429エラー: 60秒待機 ({attempt+1}/3)")
                time.sleep(60)
                continue
            logger.warning(f"note記事生成失敗: {e}")
            if attempt == 2:
                logger.debug(traceback.format_exc())
            return ""

    return ""


def generate_all_reports() -> list[Path]:
    """全重賞レースのnote記事を生成して保存する。"""
    cache_path = DATA_DIR / "predictions_cache.json"
    if not cache_path.exists():
        logger.warning("predictions_cache.json が見つかりません")
        return []

    with open(cache_path, encoding="utf-8") as f:
        cache = json.load(f)

    output_dir = DATA_DIR / "note_articles"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for race_id, entry in cache.items():
        if race_id.startswith("_"):
            continue
        if entry.get("is_grade") is False:
            continue  # 平場はスキップ

        race_name = entry.get("race_name", race_id)
        logger.info(f"note記事生成中: {race_name}")

        report = generate_analyst_report(race_id, cache)
        if not report:
            continue

        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in race_name)
        path = output_dir / f"{safe_name}.md"
        path.write_text(report, encoding="utf-8")
        saved.append(path)
        logger.info(f"保存: {path}")

    return saved


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    paths = generate_all_reports()
    print(f"生成完了: {len(paths)} 記事")
    for p in paths:
        print(f"  {p}")
