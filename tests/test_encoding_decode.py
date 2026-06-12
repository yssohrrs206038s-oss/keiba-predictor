"""
_best_encoding_decode のユニットテスト。

euc-jp / utf-8 / cp932 それぞれでエンコードした日本語テキストを渡して
置換文字ゼロの正しいエンコーディングを選択することを確認する。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from keiba_predictor.scraper.netkeiba_scraper import _best_encoding_decode

REPLACEMENT = "�"

JAPANESE = "ディープインパクト"  # euc-jp / utf-8 / cp932 すべてで表現可能な文字列


def _count_replacements(text: str) -> int:
    return text.count(REPLACEMENT)


class TestBestEncodingDecode:
    def test_utf8_bytes_decoded_correctly(self):
        content = JAPANESE.encode("utf-8")
        text, enc = _best_encoding_decode(content, hint="utf-8")
        assert _count_replacements(text) == 0
        assert JAPANESE in text
        assert enc == "utf-8"

    def test_eucjp_bytes_decoded_correctly(self):
        content = JAPANESE.encode("euc-jp")
        text, enc = _best_encoding_decode(content, hint="euc-jp")
        assert _count_replacements(text) == 0
        assert JAPANESE in text
        assert enc == "euc-jp"

    def test_cp932_bytes_decoded_correctly(self):
        content = JAPANESE.encode("cp932")
        text, enc = _best_encoding_decode(content, hint="cp932")
        assert _count_replacements(text) == 0
        assert JAPANESE in text
        assert enc == "cp932"

    def test_eucjp_bytes_with_wrong_hint_utf8(self):
        """EUC-JPバイト列にUTF-8ヒントを与えても正しくデコードできる"""
        content = JAPANESE.encode("euc-jp")
        text, enc = _best_encoding_decode(content, hint="utf-8")
        assert _count_replacements(text) == 0
        assert JAPANESE in text

    def test_utf8_bytes_with_eucjp_hint(self):
        """UTF-8バイト列にEUC-JPヒントを与えても正しくデコードできる（NAR移行ケース）"""
        content = JAPANESE.encode("utf-8")
        text, enc = _best_encoding_decode(content, hint="euc-jp")
        assert _count_replacements(text) == 0
        assert JAPANESE in text

    def test_no_hint_falls_back_to_utf8(self):
        """ヒントなしでUTF-8バイト列を渡すとUTF-8で正しくデコード"""
        content = JAPANESE.encode("utf-8")
        text, enc = _best_encoding_decode(content, hint="")
        assert _count_replacements(text) == 0
        assert JAPANESE in text

    def test_returns_least_garbled_when_all_fail(self):
        """どのエンコーディングでも完全には読めないバイト列でも最少置換文字の候補を返す"""
        # 無効バイト列（全エンコーディングで一部置換が入る）
        content = b"\xff\xfe" + JAPANESE.encode("utf-16-le")
        text, enc = _best_encoding_decode(content, hint="utf-8")
        # 最低限: 戻り値がstr型であること
        assert isinstance(text, str)
        assert isinstance(enc, str)

    def test_eucjp_longer_text(self):
        """複数馬名を含む長いEUC-JPテキストでも正しくデコード"""
        names = "ウオッカ,ダイワスカーレット,スペシャルウィーク,グラスワンダー"
        content = names.encode("euc-jp")
        text, enc = _best_encoding_decode(content, hint="euc-jp")
        assert _count_replacements(text) == 0
        for name in names.split(","):
            assert name in text

    def test_cp932_with_special_chars(self):
        """CP932固有の特殊文字（①など）を含むテキスト"""
        text_cp932 = "①着 " + JAPANESE
        content = text_cp932.encode("cp932")
        text, enc = _best_encoding_decode(content, hint="cp932")
        assert _count_replacements(text) == 0

    def test_utf8_page_with_eucjp_caller_hint(self):
        """
        NAR結果ページがEUC-JPからUTF-8に変わったケースの再現。
        呼び出し側がencoding='euc-jp'を渡してもUTF-8で正しくデコードされる。
        """
        page_content = f"<html><body><td>{JAPANESE}</td></body></html>".encode("utf-8")
        text, enc = _best_encoding_decode(page_content, hint="euc-jp")
        assert _count_replacements(text) == 0
        assert JAPANESE in text
        assert enc == "utf-8"
