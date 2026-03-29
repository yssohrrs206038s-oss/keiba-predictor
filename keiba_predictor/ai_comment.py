"""
Claude API（anthropic SDK）を使った予測結果の自然言語解説生成モジュール。

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
import time
import traceback

# ログ末尾に一括表示するためのバッファ
_pending_reports: list[tuple[str, str]] = []  # [(race_name, text), ...]
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── 使用するモデル ─────────────────────────────────────────────
MODEL_ID = "claude-haiku-4-5-20251001"

# ── 1頭あたりの最大解説文字数（Discord の行幅に合わせて調整） ──
MAX_COMMENT_LEN = 150


# ══════════════════════════════════════════════════════════════
# Windows 互換出力ヘルパー
# ══════════════════════════════════════════════════════════════

def _setup_utf8_stdout() -> None:
    """Windows で stdout/stderr を UTF-8 に強制する。"""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
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
        _p(f"  [AI ERROR] {msg}")

    def _dbg(msg: str) -> None:
        """デバッグ出力。verbose時のみprint、それ以外はlogger.debug。"""
        logger.debug(f"[generate_comments] {msg}")
        if verbose:
            _p(f"  [AI] {msg}")

    # ── Step 1: API キー確認 ─────────────────────────────────
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        _dbg("ANTHROPIC_API_KEY が設定されていないためスキップ")
        return {}
    _dbg(f"Step1 OK: API キー末尾6桁=...{key[-6:]}")

    # ── Step 2: anthropic インポート ─────────────────────────
    try:
        import anthropic
        _dbg(f"Step2 OK: anthropic インポート成功")
    except ImportError as e:
        _err(f"anthropic パッケージが未インストールです: {e}")
        return {}

    # ── Step 3〜6: データ組立 → APIコール → JSON解析 ────────
    # 全体をtry/exceptで囲み、予期しないエラーでもクラッシュしない
    try:
        return _generate_comments_inner(result_df, race_name, course_info, key, _dbg, _err, _log)
    except Exception as e:
        _err(f"AI解説生成で予期しないエラー: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        return {}


def _generate_comments_inner(
    result_df: pd.DataFrame,
    race_name: str,
    course_info: str,
    key: str,
    _dbg,
    _err,
    _log,
) -> dict[str, str]:
    """generate_comments の内部実装。エラーは呼び出し元で捕捉する。"""
    import anthropic

    _dbg(f"Step3: データ組み立て開始  race_name={race_name!r}  course_info={course_info!r}")

    # prob_top3 降順でランク付け（1位=◎🔥 2位=○✨ 3位=▲⚡）
    sorted_idx = result_df["prob_top3"].rank(ascending=False, method="first")
    MARKS = {1: "◎🔥", 2: "○✨", 3: "▲⚡"}

    horses_data = []
    for _, row in result_df.iterrows():
        rank = int(sorted_idx.loc[row.name]) if row.name in sorted_idx.index else 99
        if rank > 3:
            continue  # ◎○▲の3頭のみ処理

        num = str(int(row["horse_number"])) if pd.notna(row.get("horse_number")) else "?"

        prob = float(row["prob_top3"]) if pd.notna(row.get("prob_top3")) else 0.0
        _ev_raw = row.get("ev_score") if "ev_score" in row.index else None
        ev_val = float(_ev_raw) if pd.notna(_ev_raw) else 0.0

        entry: dict = {
            "馬番": num,
            "馬名": str(row.get("horse_name", "不明"))[:12],
            "AI印": MARKS[rank],
            "AI3着以内確率": f"{prob * 100:.1f}%",
            "EVスコア": f"{ev_val:.2f}",
            "人気": str(int(row["popularity"])) + "番人気" if pd.notna(row.get("popularity")) else "?",
        }

        pfp = pd.to_numeric(row.get("prev_finish_pos"), errors="coerce")
        if pd.notna(pfp):
            entry["前走着順"] = int(pfp)

        jfr = pd.to_numeric(row.get("jockey_fukusho_rate"), errors="coerce")
        if pd.notna(jfr):
            entry["騎手複勝率"] = f"{jfr:.3f}"

        # 脚質
        rs = pd.to_numeric(row.get("running_style_enc"), errors="coerce")
        if pd.notna(rs):
            _RS_MAP = {0: "逃げ", 1: "先行", 2: "差し", 3: "追込"}
            entry["脚質"] = _RS_MAP.get(int(rs), "不明")

        # 馬場状態適性
        tcr = pd.to_numeric(row.get("horse_track_fukusho_rate"), errors="coerce")
        if pd.notna(tcr):
            entry["馬場適性複勝率"] = f"{tcr:.3f}"

        # 期待値の妙味判定
        if ev_val >= 3.0:
            entry["期待値評価"] = "妙味あり"
        elif ev_val >= 1.5:
            entry["期待値評価"] = "標準"

        # 人気薄×高確率 = 穴馬候補
        _pop = pd.to_numeric(row.get("popularity"), errors="coerce")
        if pd.notna(_pop) and _pop >= 6 and prob >= 0.35:
            entry["穴馬候補"] = True

        # SHAP値による予測根拠を追加
        shap_top = row.get("shap_top")
        if isinstance(shap_top, list) and shap_top:
            strengths = [s["label"] for s in shap_top if s.get("value", 0) > 0]
            concerns = [s["label"] for s in shap_top if s.get("value", 0) < 0]
            if strengths:
                entry["AI判定の強み"] = "、".join(strengths)
            if concerns:
                entry["AI判定の懸念"] = "、".join(concerns)

        horses_data.append(entry)

    _dbg(f"Step3 OK: {len(horses_data)} 頭分データ組み立て完了（◎○▲のみ）")

    # ── Step 4: プロンプト生成 ────────────────────────────────
    race_label = race_name or "今回のレース"
    if course_info:
        race_label += f"（{course_info}）"

    system_prompt = (
        "あなたは「データ・展開・馬場・騎手・オッズ」を網羅したトップ競馬アナリストです。\n"
        "2026年現在の日本競馬のトレンド（高速馬場化、特定の種牡馬の傾向）を踏まえ、\n"
        "長期回収率を最大化するための「期待値重視」の分析を行います。\n\n"
        "【重要ルール】\n"
        "・提供されたデータの数値のみを使うこと\n"
        "・データにない数値（オッズ・配当金額）は絶対に書かないこと\n"
        "・語尾は断定的・自信ありげに。絵文字を効果的に使用\n"
        "・馬名は解説テキスト内に含めないこと（見出しに別途表示）"
    )

    prompt = (
        f"{race_label}の本命◎○▲3頭のAI予測データです。\n\n"
        f"{json.dumps(horses_data, ensure_ascii=False)}\n\n"
        f"各馬について{MAX_COMMENT_LEN}文字以内で解説してください。\n\n"
        f"【必須要素】\n"
        f"1. AI印（◎🔥/○✨/▲⚡）を文頭に付ける\n"
        f"2. 展開予測：脚質から展開を予測し有利/不利を言及\n"
        f"3. 馬場適性：馬場状態やコース特性との相性\n"
        f"4. 騎手評価：騎手複勝率を活用した評価\n"
        f"5. 期待値評価：「期待値評価」が「妙味あり」なら明記、「穴馬候補」があれば強調\n"
        f"6. 懸念点：「AI判定の懸念」があれば最後に1つだけ言及\n\n"
        f"【禁止】\n"
        f"・馬名を含めない\n"
        f"・オッズや配当金額\n"
        f"・データにない数値の捏造\n\n"
        f"出力：JSONのみ（コードブロック不要）\n"
        f"キー：馬番（文字列）、値：解説テキスト"
    )
    _dbg(f"Step4 OK: プロンプト {len(prompt)} 文字")

    # ── Step 5: API コール（429対策: sleep + 最大3回リトライ）────
    raw = ""
    client = anthropic.Anthropic(api_key=key)
    last_exc: Exception = Exception("未実行")
    for attempt in range(3):
        try:
            if attempt > 0:
                wait = 10 * attempt  # 10s, 20s
                _dbg(f"Step5: リトライ {attempt}/2  {wait}秒待機中...")
                time.sleep(wait)
            _dbg(f"Step5: API 呼び出し (attempt={attempt+1}/3, model={MODEL_ID!r})")
            try:
                response = client.messages.create(
                    model=MODEL_ID,
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.content[0].text
                logger.debug(f"Claude response: {len(response_text)} chars")
            except Exception as api_exc:
                logger.debug(f"Claude API Error: {type(api_exc).__name__}: {api_exc}")
                raise  # 上位の except に委譲
            raw = response_text.strip()
            _dbg(f"Step5 OK: API 応答 {len(raw)} 文字")
            _dbg(f"  raw preview: {raw[:300]!r}")
            break  # 成功
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            # 404: モデル名が無効 → リトライ不要なので即終了
            if "404" in err_str or "not found" in err_str:
                _err(f"Step5 404エラー: モデル名 {MODEL_ID!r} が無効である可能性があります")
                _err(f"  有効なモデル例: 'claude-haiku-4-5' / 'claude-sonnet-4-5'")
                _err(f"  原文: {e}")
                return {}
            # 429: レート制限 → 60秒待機してリトライ
            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                logger.warning(f"[Claude] 429エラー: 60秒待機してリトライ ({attempt+1}/3)")
                time.sleep(60)
                if attempt < 2:
                    continue
                logger.warning("[Claude] 3回失敗 スキップします")
                return {}
            _err(f"Step5 attempt {attempt+1} 失敗: {type(e).__name__}: {e}")
            if attempt == 2:
                _err(f"  traceback:\n{traceback.format_exc()}")
                return {}
        else:
            if raw and attempt > 0:
                logger.info("[Claude] リトライ成功")

    if not raw:
        _err(f"Step5: レスポンスが空  last_exc={last_exc}")
        return {}

    # ── Step 6: JSON 解析 ─────────────────────────────────────
    try:
        _dbg("Step6: JSON 解析開始")
        json_str = _extract_json_object(raw)
        _dbg(f"  json_str preview: {json_str[:200]!r}")
        comments = json.loads(json_str)
        result = {str(k): str(v)[:MAX_COMMENT_LEN] for k, v in comments.items()}
        _dbg(f"Step6 OK: {len(result)} 頭分  keys={sorted(result.keys())}")
        return result
    except json.JSONDecodeError as e:
        _err(f"Step6 JSON 解析失敗: {e}")
        _err(f"  raw (full): {raw!r}")
        return {}


# ══════════════════════════════════════════════════════════════
# レポートテキスト生成（note / BOOKERS 投稿用）
# ══════════════════════════════════════════════════════════════

def generate_report_text(
    comments_dict: dict[str, str],
    race_name: str = "",
    course_info: str = "",
    result_df: "Optional[pd.DataFrame]" = None,
    buy_lines: Optional[list] = None,
) -> str:
    """
    AI解説コメントから note / BOOKERS 投稿用のレポートテキストを生成する。

    Args:
        comments_dict : generate_comments() の返り値 {"馬番": "解説テキスト"}
        race_name     : 表示用レース名
        course_info   : コース情報（例: "中山 芝2500m"）
        result_df     : predict_race() 済みの DataFrame（馬名の取得に使用）
        buy_lines     : _build_buy_lines() の返り値（推奨買い目行リスト）

    Returns:
        整形済みレポート文字列（stdout にも print する）
    """
    SEP = "─" * 30
    header = race_name or "AI予想レポート"
    course_str = f"（{course_info}）" if course_info else ""

    lines = [
        SEP,
        f"【AI予想】{header}{course_str}",
        SEP,
        "",
    ]

    # 馬番→馬名マップを result_df から構築
    name_map: dict[str, str] = {}
    if result_df is not None and not result_df.empty:
        for _, row in result_df.iterrows():
            try:
                num = str(int(row["horse_number"]))
                name_map[num] = str(row.get("horse_name", ""))
            except Exception:
                pass

    # 馬番の数値順に詳細解説を出力（展開・血統・毒舌コメント含む）
    def _sort_key(k: str) -> int:
        try:
            return int(k)
        except ValueError:
            return 999

    for num in sorted(comments_dict.keys(), key=_sort_key):
        comment = comments_dict[num]
        horse_name = name_map.get(num, f"{num}番")
        lines.append(f"{num}番 {horse_name}")
        lines.append(f"  {comment}")
        lines.append("")

    # 推奨買い目セクション
    if buy_lines:
        lines += [SEP, "【AI推奨買い目】", SEP]
        lines += buy_lines

    lines += [
        SEP,
        "【AI解析のポイント】",
        "本レースはXGBoostによる高精度確率と、Claude Haikuによる展開・血統分析を",
        "融合した独自の期待値算出を行っています。",
        SEP,
    ]
    text = "\n".join(lines)
    return text


def save_report(text: str, race_name: str) -> "Optional[Path]":
    """
    レポートテキストを outputs/report_{race_name}.txt に保存し、
    表示用バッファ (_pending_reports) に積む。
    実際の print は flush_reports() で一括実施。

    Returns:
        保存したファイルパス。失敗時は None。
    """
    from pathlib import Path

    # ── バッファに積む（ログ末尾で一括表示するため）──────────
    _pending_reports.append((race_name, text))

    # ── ファイル保存 ─────────────────────────────────────────
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", race_name).strip() or "unknown"
    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"report_{safe_name}.txt"
    try:
        out_path.write_text(text, encoding="utf-8")
        print(f"[save_report] レポート保存: {out_path}", flush=True)
        return out_path
    except Exception as e:
        print(f"[save_report] 保存失敗: {e}", flush=True)
        return None


def flush_reports() -> None:
    """
    _pending_reports に溜まったレポートを ▼▼▼ で囲んで一括 print する。
    環境変数 DISCORD_REPORT_WEBHOOK_URL が設定されていれば自分宛チャンネルにも送信。
    predict_from_csv / predict_live の最後に呼ぶことで、
    他のログと混ざらずにまとめて表示される。
    """
    _setup_utf8_stdout()  # 絵文字等のUnicodeEncodeError対策
    if not _pending_reports:
        return

    _FENCE = "▼" * 50
    for race_name, text in _pending_reports:
        label = f"NOTE REPORT: {race_name}" if race_name else "NOTE REPORT"
        print(f"\n{_FENCE}", flush=True)
        print(f"▼▼▼  {label}  ▼▼▼", flush=True)
        print(_FENCE, flush=True)
        print(text, flush=True)
        print(_FENCE, flush=True)
        print(f"▼▼▼  END OF REPORT  ▼▼▼", flush=True)
        print(f"{_FENCE}\n", flush=True)

        # 自分宛 Discord チャンネルへ送信（DISCORD_REPORT_WEBHOOK_URL が設定されている場合のみ）
        report_url = os.environ.get("DISCORD_REPORT_WEBHOOK_URL", "")
        if not report_url:
            print("[flush_reports] DISCORD_REPORT_WEBHOOK_URL 未設定 → Discord送信スキップ", flush=True)
        else:
            print(f"[flush_reports] Sending to URL: {report_url[:40]}...", flush=True)
            _send_report_to_discord(report_url, race_name, text)

    _pending_reports.clear()


def _send_report_to_discord(webhook_url: str, race_name: str, text: str) -> None:
    """レポート全文を Discord に送信する。2000字超は自動分割。"""
    import requests

    webhook_url = webhook_url.strip()
    header = f"📋 **{race_name} レポート**\n" if race_name else "📋 **レポート**\n"
    full_text = header + text
    chunks = [full_text[i: i + 1900] for i in range(0, len(full_text), 1900)]

    for idx, chunk in enumerate(chunks):
        try:
            resp = requests.post(webhook_url, json={"content": chunk}, timeout=15)
            ok = resp.status_code in (200, 204)
            print(
                f"[flush_reports] Discord report送信 chunk {idx+1}/{len(chunks)}: "
                f"status={resp.status_code} {'✅ 成功' if ok else '⚠️ 予期しないステータス'}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[flush_reports] Discord report送信 chunk {idx+1}/{len(chunks)}: "
                f"❌ 失敗 {type(exc).__name__}: {exc}",
                flush=True,
            )
            print(traceback.format_exc(), flush=True)


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
    _setup_utf8_stdout()

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        root.addHandler(handler)

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
        _p(f"[OK] anthropic : インポート成功")
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
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with just: OK"}],
        )
        reply = resp.content[0].text.strip()
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
        verbose=True,
    )

    _p()
    if not comments:
        _p("[NG] generate_comments() returned empty dict")
        sys.exit(1)

    _p(f"[OK] generate_comments() returned {len(comments)} comments")
    _p("-" * 60)
    for num, comment in sorted(comments.items(), key=lambda x: int(x[0])):
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
            try:
                print(f"\n[FATAL] Unexpected error: {type(e).__name__}: {e}", flush=True)
            except Exception:
                pass
            sys.exit(1)
    else:
        print("Usage: python -m keiba_predictor.ai_comment --test", flush=True)
        sys.exit(1)
