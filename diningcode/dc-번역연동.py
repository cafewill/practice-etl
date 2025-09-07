#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Papago NMT 번역 유틸
- config.json에서 API 키를 읽고 번역
- 파라미터: source(기본 auto), target, text
- 실패 시 원문 text 그대로 반환
- zh, cn, tw 같은 약식 코드는 Papago 표준('zh-CN','zh-TW')으로 자동 정규화
"""

from __future__ import annotations
import json
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PAPAGO_URL = "https://papago.apigw.ntruss.com/nmt/v1/translation"


class PapagoTranslator:
    def __init__(self, config_path: str = "config.json", timeout: float = 8.0):
        self.timeout = timeout
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["POST"]),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.headers = self._load_headers(config_path)

    @staticmethod
    def _load_headers(config_path: str):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}

        client_id = cfg.get("PAPAGO_CLIENT_ID") or cfg.get("X-NCP-APIGW-API-KEY-ID")
        client_secret = cfg.get("PAPAGO_CLIENT_SECRET") or cfg.get("X-NCP-APIGW-API-KEY")
        if not client_id or not client_secret:
            # 키가 없으면 이후 translate에서 원문 반환 로직을 쓰기 위해 에러를 일으키지 않음
            # 대신 헤더를 비워두고 요청 시 실패 -> 원문 반환
            return {
                "X-NCP-APIGW-API-KEY-ID": "",
                "X-NCP-APIGW-API-KEY": "",
                "Content-Type": "application/json",
            }

        return {
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _normalize_lang(code: Optional[str]) -> str:
        if not code:
            return "en"
        c = code.replace("_", "-").lower()
        mapping = {
            "cn": "zh-CN",
            "zh": "zh-CN",
            "zh-cn": "zh-CN",
            "zh-hans": "zh-CN",
            "tw": "zh-TW",
            "zh-tw": "zh-TW",
            "zh-hant": "zh-TW",
        }
        # Papago는 대소문자 구분: 'zh-CN','zh-TW'
        normalized = mapping.get(c, code)
        if normalized.lower() in ("zh-cn", "zh_cn"):
            return "zh-CN"
        if normalized.lower() in ("zh-tw", "zh_tw"):
            return "zh-TW"
        return normalized

    def translate(self, text: str, target: str, source: str = "auto", honorific: bool = True) -> str:
        """
        번역. 실패 시 원문 반환.
        :param text: 원문 텍스트
        :param target: 타깃 언어(예: 'en', 'zh-CN', 'zh-TW', 'ja' 등)
        :param source: 소스 언어(기본 'auto')
        :param honorific: 한국어 경어 처리 여부 (Papago는 문자열 "true"/"false" 기대)
        """
        if text is None:
            return ""
        s = (source or "auto").strip() or "auto"
        t = self._normalize_lang(target)

        payload = {
            "source": s,
            "target": t,
            "text": str(text),
            "honorific": "true" if honorific else "false",
        }

        try:
            resp = self.session.post(PAPAGO_URL, headers=self.headers, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                return text  # HTTP 오류 시 원문
            data = resp.json()
            translated = (
                data.get("message", {})
                    .get("result", {})
                    .get("translatedText")
            )
            return translated if translated else text
        except Exception:
            return text


def translate_text(text: str, target: str, source: str = "auto", config_path: str = "config.json", timeout: float = 8.0) -> str:
    """
    간편 함수: 한 번 호출로 config 로딩 + 번역 수행.
    실패 시 원문 반환.
    """
    try:
        translator = PapagoTranslator(config_path=config_path, timeout=timeout)
        return translator.translate(text=text, target=target, source=source)
    except Exception:
        return text


if __name__ == "__main__":
    # 간단 테스트 CLI
    import argparse, sys
    p = argparse.ArgumentParser(description="Papago translate")
    p.add_argument("--text", required=False, help="원문 텍스트 (미지정 시 STDIN)")
    p.add_argument("--target", required=True, help="타깃 언어 (예: en, zh-CN, zh-TW, ja)")
    p.add_argument("--source", default="auto", help="소스 언어 (기본 auto)")
    p.add_argument("--config", default="config.json", help="API 키가 담긴 config.json 경로")
    args = p.parse_args()

    src_text = args.text if args.text is not None else sys.stdin.read()
    print(translate_text(src_text, target=args.target, source=args.source, config_path=args.config))
