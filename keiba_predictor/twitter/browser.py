"""
Chrome ブラウザ管理モジュール

Playwrightを使用してChromeブラウザを起動・管理する。
セッションクッキーをJSONファイルに保存して再ログインを省略できる。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# クッキー保存先
DATA_DIR = Path(__file__).parent.parent / "data"
COOKIE_FILE = DATA_DIR / "twitter_cookies.json"
STORAGE_STATE_FILE = DATA_DIR / "twitter_storage_state.json"


def get_browser_and_context(
    headless: bool = False,
    use_saved_session: bool = True,
):
    """
    Playwrightのブラウザとコンテキストを取得する。

    Args:
        headless: ヘッドレスモードで起動するか (デフォルト: False = 画面あり)
        use_saved_session: 保存済みセッションを使用するか

    Returns:
        (playwright, browser, context) のタプル
    """
    from playwright.sync_api import sync_playwright

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pw = sync_playwright().start()

    launch_args = [
        "--no-sandbox",
        "--disable-blink-features=AutomationControlled",
        "--disable-dev-shm-usage",
        "--disable-extensions",
        "--lang=ja-JP,ja",
    ]

    browser = pw.chromium.launch(
        headless=headless,
        args=launch_args,
        channel="chrome",
    )

    # 保存済みセッションを読み込む
    storage_state = None
    if use_saved_session and STORAGE_STATE_FILE.exists():
        logger.info(f"保存済みセッションを読み込み: {STORAGE_STATE_FILE}")
        storage_state = str(STORAGE_STATE_FILE)

    context = browser.new_context(
        storage_state=storage_state,
        viewport={"width": 1280, "height": 900},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        locale="ja-JP",
        timezone_id="Asia/Tokyo",
    )

    # Botと検知されにくくする
    context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
        Object.defineProperty(navigator, 'languages', {get: () => ['ja-JP', 'ja', 'en']});
        window.chrome = { runtime: {} };
    """)

    return pw, browser, context


def save_session(context) -> None:
    """現在のブラウザセッション（クッキー等）を保存する。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    context.storage_state(path=str(STORAGE_STATE_FILE))
    logger.info(f"セッションを保存しました: {STORAGE_STATE_FILE}")


def is_session_saved() -> bool:
    """保存済みセッションが存在するか確認する。"""
    return STORAGE_STATE_FILE.exists()


def clear_session() -> None:
    """保存済みセッションを削除する。"""
    if STORAGE_STATE_FILE.exists():
        STORAGE_STATE_FILE.unlink()
        logger.info("セッションを削除しました")
