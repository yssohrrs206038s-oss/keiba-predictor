"""
Claude API を使った予測結果の自然言語解説生成モジュール。

ANTHROPIC_API_KEY 未設定時はスキップして空 dict を返す（グレースフルデグラデーション）。
"""

import json
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── 使用するモデル ─────────────────────────────────────────────
MODEL_ID = "claude-opus-4-6"

# ── 1頭あたりの最大解説文字数（Discord の行幅に合わせて調整） ──
MAX_COMMENT_LEN = 80


def generate_comments(
    result_df: pd.DataFrame,
    race_name: str = "",
    course_info: str = "",
    api_key: Optional[str] = None,
) -> dict[str, str]:
    """
    各馬の解説テキストを Claude API で生成する。

    Args:
        result_df  : predict_race() + calc_ev_and_flags() 済みの DataFrame
        race_name  : 表示用レース名
        course_info: コース情報（例: "芝2500m"）
        api_key    : Anthropic API キー（省略時は環境変数 ANTHROPIC_API_KEY を使用）

    Returns:
        {"馬番(str)": "解説テキスト"} の dict。
        API キー未設定・エラー時は空 dict を返す。
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        logger.debug("ANTHROPIC_API_KEY 未設定のため AI 解説をスキップ")
        return {}

    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic パッケージが未インストールです: pip install anthropic")
        return {}

    # ── 送信する馬データを組み立て ───────────────────────────
    horses_data = []
    for _, row in result_df.iterrows():
        num = str(int(row["horse_number"])) if pd.notna(row.get("horse_number")) else "?"
        entry: dict = {
            "馬番": num,
            "馬名": str(row.get("horse_name", "不明"))[:12],
            "AI3着以内確率": f"{row['prob_top3'] * 100:.1f}%",
            "EVスコア": f"{row['ev_score']:.2f}" if pd.notna(row.get("ev_score")) else "N/A",
            "人気": str(int(row["popularity"])) if pd.notna(row.get("popularity")) else "?",
            "オッズ": str(row.get("odds", "?")),
        }

        # オプション特徴量（欠損時は省略）
        pfp = pd.to_numeric(row.get("prev_finish_pos"), errors="coerce")
        if pd.notna(pfp):
            entry["前走着順"] = int(pfp)

        prev_odds = pd.to_numeric(row.get("prev_odds"), errors="coerce")
        if pd.notna(prev_odds):
            entry["前走オッズ"] = float(prev_odds)

        jfr = pd.to_numeric(row.get("jockey_fukusho_rate"), errors="coerce")
        if pd.notna(jfr):
            entry["騎手複勝率"] = f"{jfr:.3f}"

        ctype = row.get("course_type_enc")
        if pd.notna(ctype):
            entry["コース種別"] = "芝" if int(ctype) == 0 else "ダート"

        dist_diff = pd.to_numeric(row.get("dist_diff_prev"), errors="coerce")
        if pd.notna(dist_diff):
            entry["前走距離差m"] = int(dist_diff)

        wdiff = pd.to_numeric(row.get("horse_weight_diff"), errors="coerce")
        if pd.notna(wdiff):
            entry["馬体重増減kg"] = int(wdiff)

        danger = row.get("danger_reasons", [])
        if danger:
            entry["危険フラグ"] = danger

        horses_data.append(entry)

    # ── プロンプト生成 ────────────────────────────────────────
    race_label = race_name or "今回のレース"
    if course_info:
        race_label += f"（{course_info}）"

    prompt = f"""\
あなたは競馬予測AIの解説アシスタントです。
以下は「{race_label}」の予測データです。

各馬のデータ（JSON配列）:
{json.dumps(horses_data, ensure_ascii=False, indent=2)}

上記データを基に、各馬について競馬ファン向けの自然な日本語解説を生成してください。

出力形式:
- 必ず以下のJSONオブジェクトのみを返す（コードブロック不要、JSON以外の文字列不要）
- キー: 馬番（文字列）、値: 解説テキスト（最大{MAX_COMMENT_LEN}文字、改行なし）
- 解説には「なぜその順位か」「強み・懸念点」「買い推奨/見送り理由」を簡潔に含める
- EVスコアが高い馬は配当妙味も触れる
- 危険フラグのある馬はその理由を含める

例: {{"1": "前走快勝の勢いそのまま。芝適性が高くEV良好。", "3": "1番人気だがAI確率が低く危険。前走凡走で信頼しにくい。"}}
"""

    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = next(
            (b.text for b in response.content if b.type == "text"), ""
        ).strip()

        # JSON 部分のみ抽出（余分なテキストがある場合に対応）
        if raw.startswith("{"):
            comments = json.loads(raw)
        else:
            # コードブロックを除去
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
            comments = json.loads(raw)

        # 文字数制限を適用
        return {
            str(k): str(v)[:MAX_COMMENT_LEN]
            for k, v in comments.items()
        }

    except json.JSONDecodeError as e:
        logger.warning(f"AI解説: JSON解析失敗 ({e})")
        return {}
    except Exception as e:
        logger.warning(f"AI解説生成失敗: {e}")
        return {}
