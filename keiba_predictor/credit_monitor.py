"""
Anthropic API クレジット残量監視

残量が閾値以下になったら Discord に警告通知する。

使い方:
    python -m keiba_predictor.credit_monitor
"""

import logging
import os
import sys

import requests

logger = logging.getLogger(__name__)

THRESHOLD = 1.0  # USD


def check_credit() -> None:
    """Anthropic API のクレジット残量を確認し、閾値以下なら Discord に警告する。"""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")

    if not api_key:
        logger.warning("ANTHROPIC_API_KEY が未設定です")
        return

    # Anthropic Balance API
    try:
        resp = requests.get(
            "https://api.anthropic.com/v1/organizations/balance",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"クレジット残量取得失敗: {e}")
        # APIがない場合はUsage APIを試す
        try:
            resp = requests.get(
                "https://api.anthropic.com/v1/usage",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e2:
            logger.warning(f"Usage API も失敗: {e2}")
            return

    # 残量を取得（APIレスポンス構造に応じて調整）
    balance = None
    if "balance" in data:
        balance = float(data["balance"])
    elif "credit_balance" in data:
        balance = float(data["credit_balance"])
    elif "available" in data:
        balance = float(data["available"])

    if balance is None:
        logger.info(f"クレジット残量を解析できませんでした。レスポンス: {data}")
        return

    logger.info(f"Anthropic クレジット残量: ${balance:.2f}")
    print(f"[credit] 残量: ${balance:.2f} (閾値: ${THRESHOLD:.2f})", flush=True)

    if balance <= THRESHOLD:
        msg = (
            f"⚠️ Anthropic APIクレジット残量が${THRESHOLD:.2f}を下回りました"
            f"（現在: ${balance:.2f}）。チャージを検討してください。"
        )
        logger.warning(msg)
        if webhook_url:
            try:
                requests.post(webhook_url, json={"content": msg}, timeout=15)
                logger.info("Discord 警告送信完了")
            except Exception as e:
                logger.warning(f"Discord 送信失敗: {e}")
        else:
            print(msg, flush=True)
    else:
        print(f"[credit] OK: ${balance:.2f} > ${THRESHOLD:.2f}", flush=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    check_credit()
