"""
GitHub Releases の model-latest タグからモデルファイルをダウンロードする。

使い方:
    python keiba_predictor/scripts/download_model.py

環境変数:
    GITHUB_REPOSITORY  - "owner/repo" 形式（Actions では自動設定）
    GITHUB_TOKEN       - ダウンロード認証トークン（プライベートリポジトリ時）

ダウンロード対象:
    *.pkl（keiba_predictor/model/ 以下に保存）

Releases に該当タグが存在しない場合はエラー終了する。
"""

import os
import sys
import logging
from pathlib import Path
import urllib.request
import urllib.error
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TAG = "model-latest"
MODEL_DIR = Path(__file__).parent.parent / "model"

GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "") or os.environ.get("GH_TOKEN", "")


def _api_get(url: str) -> dict:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if GITHUB_TOKEN:
        req.add_header("Authorization", f"Bearer {GITHUB_TOKEN}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/octet-stream")
    if GITHUB_TOKEN:
        req.add_header("Authorization", f"Bearer {GITHUB_TOKEN}")
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        data = resp.read()
        f.write(data)
    size_kb = dest.stat().st_size // 1024
    logger.info(f"  保存完了: {dest.name} ({size_kb} KB)")


def main() -> None:
    if not GITHUB_REPOSITORY:
        logger.error("GITHUB_REPOSITORY 環境変数が未設定です（例: owner/repo）")
        sys.exit(1)

    api_url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/releases/tags/{TAG}"
    logger.info(f"Releases タグ取得: {api_url}")

    try:
        release = _api_get(api_url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.error(
                f"Releases タグ '{TAG}' が見つかりません。"
                f" リポジトリ={GITHUB_REPOSITORY}"
            )
        else:
            logger.error(f"GitHub API エラー: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"接続エラー: {e}")
        sys.exit(1)

    assets = release.get("assets", [])
    pkl_assets = [a for a in assets if a["name"].endswith(".pkl")]
    if not pkl_assets:
        logger.error(f"Releases '{TAG}' に .pkl ファイルが存在しません")
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for asset in pkl_assets:
        name = asset["name"]
        download_url = asset["browser_download_url"]
        dest = MODEL_DIR / name
        logger.info(f"ダウンロード: {name} ({asset['size'] // 1024} KB) ← {download_url}")
        try:
            _download(download_url, dest)
        except Exception as e:
            logger.error(f"ダウンロード失敗: {name}: {e}")
            sys.exit(1)

    logger.info(f"✅ モデルダウンロード完了: {len(pkl_assets)} ファイル")


if __name__ == "__main__":
    main()
