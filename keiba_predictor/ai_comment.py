"""
Claude API を使った予測結果の自然言語解説生成モジュール。

ANTHROPIC_API_KEY 未設定時はスキップして空 dict を返す（グレースフルデグラデーション）。

CLI テスト:
    python -m keiba_predictor.ai_comment --test
"""

import json
import logging
import os
import re
import sys
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── 使用するモデル ─────────────────────────────────────────────
MODEL_ID = "claude-opus-4-6"

# ── 1頭あたりの最大解説文字数（Discord の行幅に合わせて調整） ──
MAX_COMMENT_LEN = 80


def _extract_json_object(text: str) -> str:
    """レスポンステキストから JSON オブジェクト部分のみを抽出する。"""
    text = text.strip()
    # コードブロック (```json ... ``` or ``` ... ```) を除去
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # { ... } の範囲を正規表現で抽出（入れ子に対応）
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


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
        logger.info("AI解説スキップ: ANTHROPIC_API_KEY が設定されていません")
        return {}

    logger.info(f"AI解説生成開始: {race_name or '（レース名未指定）'} / モデル={MODEL_ID}")

    try:
        import anthropic
        logger.info(f"anthropic バージョン: {anthropic.__version__}")
    except ImportError:
        logger.error("anthropic パッケージが未インストールです: pip install anthropic")
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

    logger.info(f"  送信頭数: {len(horses_data)} 頭")

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
- EVスコアが高い馬は配当妙味にも触れる
- 危険フラグのある馬はその理由を含める

例: {{"1": "前走快勝の勢いそのまま。芝適性が高くEV良好。", "3": "1番人気だがAI確率が低く危険。前走凡走で信頼しにくい。"}}
"""

    try:
        logger.info(f"  Claude API 呼び出し中 (model={MODEL_ID})...")
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = next(
            (b.text for b in response.content if b.type == "text"), ""
        ).strip()
        logger.info(f"  API 応答取得: {len(raw)} 文字 (stop_reason={response.stop_reason})")
        logger.debug(f"  生レスポンス: {raw[:200]}")

        json_str = _extract_json_object(raw)
        comments = json.loads(json_str)
        result = {str(k): str(v)[:MAX_COMMENT_LEN] for k, v in comments.items()}
        logger.info(f"  AI解説生成完了: {len(result)} 頭分")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"AI解説: JSON解析失敗 ({e})\n  raw={raw[:300]!r}")
        return {}
    except Exception as e:
        logger.error(f"AI解説生成失敗: {type(e).__name__}: {e}")
        return {}


# ══════════════════════════════════════════════════════════════
# CLI テスト: python -m keiba_predictor.ai_comment --test
# ══════════════════════════════════════════════════════════════

def _make_test_df() -> pd.DataFrame:
    """テスト用のダミー DataFrame を返す。"""
    return pd.DataFrame([
        {
            "horse_number": 1, "horse_name": "テストウマA", "prob_top3": 0.673,
            "ev_score": 4.98, "popularity": 2, "odds": 7.4,
            "prev_finish_pos": 1.0, "prev_odds": 5.2,
            "jockey_fukusho_rate": 0.312, "course_type_enc": 0,
            "dist_diff_prev": 0.0, "horse_weight_diff": -2.0,
            "is_dangerous": False, "danger_reasons": [],
        },
        {
            "horse_number": 2, "horse_name": "テストウマB", "prob_top3": 0.502,
            "ev_score": 2.76, "popularity": 1, "odds": 5.5,
            "prev_finish_pos": 5.0, "prev_odds": 3.1,
            "jockey_fukusho_rate": 0.289, "course_type_enc": 0,
            "dist_diff_prev": 200.0, "horse_weight_diff": 4.0,
            "is_dangerous": True,
            "danger_reasons": ["1〜2番人気だが前走5着"],
        },
        {
            "horse_number": 3, "horse_name": "テストウマC", "prob_top3": 0.312,
            "ev_score": 1.87, "popularity": 3, "odds": 6.0,
            "prev_finish_pos": 2.0, "prev_odds": 8.4,
            "jockey_fukusho_rate": 0.255, "course_type_enc": 0,
            "dist_diff_prev": -200.0, "horse_weight_diff": 0.0,
            "is_dangerous": False, "danger_reasons": [],
        },
    ])


def _run_test() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    print("=" * 60)
    print("AI解説 単体テスト")
    print("=" * 60)
    print(f"ANTHROPIC_API_KEY: {'設定済み (末尾6桁: ...{})'.format(key[-6:]) if key else '未設定 ❌'}")
    print(f"使用モデル       : {MODEL_ID}")
    print()

    if not key:
        print("⚠️  ANTHROPIC_API_KEY が設定されていないためAPIコールをスキップします。")
        print("   export ANTHROPIC_API_KEY=sk-ant-... を実行してから再テストしてください。")
        sys.exit(1)

    # anthropic インポート確認
    try:
        import anthropic
        print(f"✅ anthropic インポート OK (v{anthropic.__version__})")
    except ImportError as e:
        print(f"❌ anthropic インポート失敗: {e}")
        print("   pip install anthropic を実行してください。")
        sys.exit(1)

    # API 疎通確認（最小コール）
    print("\n[1] API 疎通確認中...")
    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=MODEL_ID,
            max_tokens=16,
            messages=[{"role": "user", "content": "こんにちは。「OK」とだけ返してください。"}],
        )
        reply = next((b.text for b in resp.content if b.type == "text"), "")
        print(f"✅ API 疎通 OK (応答: {reply!r})")
    except Exception as e:
        print(f"❌ API 疎通失敗: {type(e).__name__}: {e}")
        sys.exit(1)

    # generate_comments() テスト
    print("\n[2] generate_comments() テスト中...")
    test_df = _make_test_df()
    print(f"   テストデータ: {len(test_df)} 頭")
    comments = generate_comments(
        test_df,
        race_name="テストレース",
        course_info="芝2000m",
        api_key=key,
    )

    if not comments:
        print("❌ generate_comments() が空 dict を返しました")
        sys.exit(1)

    print(f"\n✅ AI解説生成成功 ({len(comments)} 頭分)")
    print("-" * 60)
    for num, comment in sorted(comments.items(), key=lambda x: int(x[0])):
        print(f"  馬番{num}: {comment}")
    print("-" * 60)
    print("\n✅ 全テスト通過")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_test()
    else:
        print("使用方法: python -m keiba_predictor.ai_comment --test")
        sys.exit(1)
