"""
Claude API を使った予測結果の自然言語解説生成モジュール。

ANTHROPIC_API_KEY 未設定時はスキップして空 dict を返す（グレースフルデグラデーション）。

CLI テスト:
    python -m keiba_predictor.ai_comment --test

Windows コマンドプロンプトで環境変数を設定する方法:
    set ANTHROPIC_API_KEY=sk-ant-...
    python -m keiba_predictor.ai_comment --test

PowerShell の場合:
    $env:ANTHROPIC_API_KEY="sk-ant-..."
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
MODEL_ID = "claude-haiku-4-5-20251001"

# ── 1頭あたりの最大解説文字数（Discord の行幅に合わせて調整） ──
MAX_COMMENT_LEN = 80


# ══════════════════════════════════════════════════════════════
# Windows 互換出力ヘルパー
# ══════════════════════════════════════════════════════════════

def _setup_utf8_stdout() -> None:
    """Windows で stdout/stderr を UTF-8 に強制する。"""
    if sys.platform == "win32":
        try:
            # Python 3.7+ — reconfigure でエンコードを変更
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            # fallback: io.TextIOWrapper で置き換え
            import io
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace"
            )


def _p(msg: str = "") -> None:
    """エンコードエラーを握り潰して stdout に flush 出力する。"""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        # 表示できない文字を ? に置換してフォールバック出力
        safe = msg.encode(sys.stdout.encoding or "ascii", errors="replace").decode(
            sys.stdout.encoding or "ascii", errors="replace"
        )
        print(safe, flush=True)


# ══════════════════════════════════════════════════════════════
# JSON 抽出ヘルパー
# ══════════════════════════════════════════════════════════════

def _extract_json_object(text: str) -> str:
    """レスポンステキストから JSON オブジェクト部分のみを抽出する。"""
    text = text.strip()
    # コードブロック (```json ... ``` or ``` ... ```) を除去
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # { ... } の範囲を深さ追跡で抽出（入れ子対応）
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


# ══════════════════════════════════════════════════════════════
# メイン公開関数
# ══════════════════════════════════════════════════════════════

def generate_comments(
    result_df: pd.DataFrame,
    race_name: str = "",
    course_info: str = "",
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> dict[str, str]:
    """
    各馬の解説テキストを Claude API で生成する。

    Args:
        result_df  : predict_race() + calc_ev_and_flags() 済みの DataFrame
        race_name  : 表示用レース名
        course_info: コース情報（例: "芝2500m"）
        api_key    : Anthropic API キー（省略時は環境変数 ANTHROPIC_API_KEY を使用）
        verbose    : True のとき print() で進捗を逐次出力する（--test 時に使用）

    Returns:
        {"馬番(str)": "解説テキスト"} の dict。
        API キー未設定・エラー時は空 dict を返す。
    """

    def _log(msg: str) -> None:
        logger.info(msg)
        if verbose:
            _p(f"  [AI] {msg}")

    def _err(msg: str) -> None:
        logger.error(msg)
        if verbose:
            _p(f"  [AI ERROR] {msg}")

    # ── Step 1: API キー確認 ─────────────────────────────────
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        _log("ANTHROPIC_API_KEY が設定されていないため AI 解説をスキップします")
        return {}
    _log(f"API キー確認 OK (末尾6桁: ...{key[-6:]})")

    # ── Step 2: anthropic インポート ─────────────────────────
    try:
        import anthropic as _anthropic
        _log(f"anthropic バージョン: {_anthropic.__version__}")
    except ImportError:
        _err("anthropic パッケージが未インストールです: pip install anthropic")
        return {}

    # ── Step 3: 馬データを組み立て ──────────────────────────
    _log(f"解説生成開始: {race_name or '（レース名未指定）'}")
    horses_data = []
    for _, row in result_df.iterrows():
        num = str(int(row["horse_number"])) if pd.notna(row.get("horse_number")) else "?"
        entry: dict = {
            "馬番": num,
            "馬名": str(row.get("horse_name", "不明"))[:12],
            "AI3着以内確率": f"{row['prob_top3'] * 100:.1f}%",
            "EVスコア": (
                f"{row['ev_score']:.2f}" if pd.notna(row.get("ev_score")) else "N/A"
            ),
            "人気": str(int(row["popularity"])) if pd.notna(row.get("popularity")) else "?",
            "オッズ": str(row.get("odds", "?")),
        }

        for col, key_jp in [
            ("prev_finish_pos", "前走着順"),
            ("prev_odds",       "前走オッズ"),
        ]:
            val = pd.to_numeric(row.get(col), errors="coerce")
            if pd.notna(val):
                entry[key_jp] = float(val)

        jfr = pd.to_numeric(row.get("jockey_fukusho_rate"), errors="coerce")
        if pd.notna(jfr):
            entry["騎手複勝率"] = f"{jfr:.3f}"

        ctype = row.get("course_type_enc")
        if pd.notna(ctype):
            entry["コース種別"] = "芝" if int(ctype) == 0 else "ダート"

        for col, key_jp in [
            ("dist_diff_prev",    "前走距離差m"),
            ("horse_weight_diff", "馬体重増減kg"),
        ]:
            val = pd.to_numeric(row.get(col), errors="coerce")
            if pd.notna(val):
                entry[key_jp] = int(val)

        danger = row.get("danger_reasons", [])
        if danger:
            entry["危険フラグ"] = danger

        horses_data.append(entry)

    _log(f"送信頭数: {len(horses_data)} 頭")

    # ── Step 4: プロンプト生成 ────────────────────────────────
    race_label = race_name or "今回のレース"
    if course_info:
        race_label += f"（{course_info}）"

    prompt = (
        f"あなたは競馬予測AIの解説アシスタントです。\n"
        f"以下は「{race_label}」の予測データです。\n\n"
        f"各馬のデータ（JSON配列）:\n"
        f"{json.dumps(horses_data, ensure_ascii=False, indent=2)}\n\n"
        f"上記データを基に、各馬について競馬ファン向けの自然な日本語解説を生成してください。\n\n"
        f"出力形式:\n"
        f"- 必ず以下のJSONオブジェクトのみを返す（コードブロック不要、JSON以外の文字列不要）\n"
        f"- キー: 馬番（文字列）、値: 解説テキスト（最大{MAX_COMMENT_LEN}文字、改行なし）\n"
        f"- 解説には「なぜその順位か」「強み・懸念点」「買い推奨/見送り理由」を簡潔に含める\n"
        f"- EVスコアが高い馬は配当妙味にも触れる\n"
        f"- 危険フラグのある馬はその理由を含める\n\n"
        f'例: {{"1": "前走快勝の勢いそのまま。芝適性高くEV良好。", '
        f'"3": "1番人気だがAI確率低く危険。前走凡走で信頼しにくい。"}}\n'
    )

    # ── Step 5: API コール ────────────────────────────────────
    raw = ""
    try:
        _log(f"Claude API 呼び出し中 (model={MODEL_ID})...")
        client = _anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = next(
            (b.text for b in response.content if b.type == "text"), ""
        ).strip()
        _log(f"API 応答取得: {len(raw)} 文字 (stop_reason={response.stop_reason})")

        # ── Step 6: JSON 解析 ─────────────────────────────────
        json_str = _extract_json_object(raw)
        comments = json.loads(json_str)
        result = {str(k): str(v)[:MAX_COMMENT_LEN] for k, v in comments.items()}
        _log(f"AI 解説生成完了: {len(result)} 頭分")
        return result

    except json.JSONDecodeError as e:
        _err(f"JSON 解析失敗: {e} | raw={raw[:200]!r}")
        return {}
    except Exception as e:
        _err(f"API 呼び出し失敗: {type(e).__name__}: {e}")
        return {}


# ══════════════════════════════════════════════════════════════
# CLI テスト: python -m keiba_predictor.ai_comment --test
# ══════════════════════════════════════════════════════════════

def _make_test_df() -> pd.DataFrame:
    """テスト用のダミー DataFrame を返す。"""
    return pd.DataFrame([
        {
            "horse_number": 1, "horse_name": "TestHorseA", "prob_top3": 0.673,
            "ev_score": 4.98, "popularity": 2, "odds": 7.4,
            "prev_finish_pos": 1.0, "prev_odds": 5.2,
            "jockey_fukusho_rate": 0.312, "course_type_enc": 0,
            "dist_diff_prev": 0.0, "horse_weight_diff": -2.0,
            "is_dangerous": False, "danger_reasons": [],
        },
        {
            "horse_number": 2, "horse_name": "TestHorseB", "prob_top3": 0.502,
            "ev_score": 2.76, "popularity": 1, "odds": 5.5,
            "prev_finish_pos": 5.0, "prev_odds": 3.1,
            "jockey_fukusho_rate": 0.289, "course_type_enc": 0,
            "dist_diff_prev": 200.0, "horse_weight_diff": 4.0,
            "is_dangerous": True,
            "danger_reasons": ["1~2ban-ninkidaga maesou 5chakujun"],
        },
        {
            "horse_number": 3, "horse_name": "TestHorseC", "prob_top3": 0.312,
            "ev_score": 1.87, "popularity": 3, "odds": 6.0,
            "prev_finish_pos": 2.0, "prev_odds": 8.4,
            "jockey_fukusho_rate": 0.255, "course_type_enc": 0,
            "dist_diff_prev": -200.0, "horse_weight_diff": 0.0,
            "is_dangerous": False, "danger_reasons": [],
        },
    ])


def _run_test() -> None:
    # ── stdout を UTF-8 に強制（Windows 対策）────────────────
    _setup_utf8_stdout()

    # ── logging を stdout へ（stderr だと cmd.exe では見えない場合がある）
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        root.addHandler(handler)

    # ── ヘッダー（ASCII のみ — エンコード問題が起きても最初の行は必ず出る）
    _p("=" * 60)
    _p("KEIBA EDGE / AI Comment Module -- Self Test")
    _p("=" * 60)
    _p(f"Python   : {sys.version}")
    _p(f"Platform : {sys.platform}")
    _p(f"Encoding : stdout={getattr(sys.stdout, 'encoding', 'unknown')}"
       f"  stderr={getattr(sys.stderr, 'encoding', 'unknown')}")
    _p(f"Model    : {MODEL_ID}")
    _p()

    # ── 環境変数確認 ─────────────────────────────────────────
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        _p(f"[OK] ANTHROPIC_API_KEY : set (last 6 chars: ...{key[-6:]})")
    else:
        _p("[NG] ANTHROPIC_API_KEY : NOT SET")
        _p()
        _p("  Windows CMD    : set ANTHROPIC_API_KEY=sk-ant-...")
        _p("  PowerShell     : $env:ANTHROPIC_API_KEY='sk-ant-...'")
        _p("  Git Bash/Linux : export ANTHROPIC_API_KEY=sk-ant-...")
        _p()
        _p("Set the key and re-run:  python -m keiba_predictor.ai_comment --test")
        sys.exit(1)

    # ── anthropic インポート確認 ──────────────────────────────
    try:
        import anthropic
        _p(f"[OK] anthropic         : v{anthropic.__version__}")
    except ImportError as e:
        _p(f"[NG] anthropic import failed: {e}")
        _p("     Run: pip install anthropic")
        sys.exit(1)

    # ── Step 1: API 疎通確認（最小コール）────────────────────
    _p()
    _p("[Step 1] API connectivity check ...")
    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=MODEL_ID,
            max_tokens=10,
            messages=[{"role": "user", "content": "Reply with just: OK"}],
        )
        reply = next((b.text for b in resp.content if b.type == "text"), "(empty)")
        _p(f"  [OK] API response: {reply!r}")
    except Exception as e:
        _p(f"  [NG] API call failed: {type(e).__name__}: {e}")
        sys.exit(1)

    # ── Step 2: generate_comments() エンドツーエンド ─────────
    _p()
    _p("[Step 2] generate_comments() end-to-end test ...")
    test_df = _make_test_df()
    _p(f"  Test data: {len(test_df)} horses")
    comments = generate_comments(
        test_df,
        race_name="TestRace",
        course_info="Turf2000m",
        api_key=key,
        verbose=True,     # <- 進捗を print で表示
    )

    _p()
    if not comments:
        _p("[NG] generate_comments() returned empty dict")
        sys.exit(1)

    _p(f"[OK] generate_comments() returned {len(comments)} comments")
    _p("-" * 60)
    for num, comment in sorted(comments.items(), key=lambda x: int(x[0])):
        # 出力時もエンコード安全に
        _p(f"  Horse #{num}: {comment}")
    _p("-" * 60)
    _p()
    _p("[PASS] All tests passed.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        try:
            _run_test()
        except SystemExit:
            raise
        except Exception as e:
            # 予期しない例外をキャッチして確実に表示する
            try:
                print(f"\n[FATAL] Unexpected error: {type(e).__name__}: {e}", flush=True)
            except Exception:
                pass
            sys.exit(1)
    else:
        print("Usage: python -m keiba_predictor.ai_comment --test", flush=True)
        sys.exit(1)
