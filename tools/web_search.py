"""
tools/web_search.py
===================
工具③：简单网络搜索。

设计原理：
    - 封装「搜索 → 取前 K 条结果」的通用能力，支持多种后端：
        * bing：必应（cn.bing.com）搜索结果页解析——国内直连可达、无需 Key，
                是本项目默认后端（兼顾国内网络环境）；
        * wikipedia：维基百科搜索 API（海外环境可达）；
        * duckduckgo：DuckDuckGo Instant Answer API（海外环境可达）。
    - 工程防御：
        * requests 超时（AGENT_SEARCH_TIMEOUT），防止外部服务无响应挂死进程；
        * 失败自动重试（最多 2 次，指数退避），容忍瞬时网络抖动；
        * 识别搜索引擎反爬/验证页，明确报错而非返回乱码；
        * 任何失败统一抛 SearchError，由 ReAct 循环反馈给模型继续规划。
    - 网络可达性因环境而异：默认 bing 国内可用；若切换后端不可达，
      可配置 AGENT_SEARCH_ENGINE 更换。
"""

from __future__ import annotations

import base64
import html as html_lib
import json
import re
import time
import urllib.parse
from typing import Any

import requests
from langchain_core.tools import tool

from config.settings import get_settings
from core.exceptions import SearchError
from core.logging import get_logger

logger = get_logger(__name__)

# 请求头：伪装常见 UA，降低被目标站拒绝的概率
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}

# 重试策略：次数与退避间隔（秒）
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 0.5


def _decode_bing_url(ck_url: str) -> str:
    """解码必应重定向 URL（形如 https://cn.bing.com/ck/a?...&u=<base64>）。

    Bing 结果页的链接指向自己的跳转地址，真实目标 URL 在 u= 参数（base64）。
    解码失败时原样返回（不影响摘要信息）。
    """
    parsed = urllib.parse.urlparse(ck_url)
    params = urllib.parse.parse_qs(parsed.query)
    encoded = params.get("u", [""])[0]
    if encoded:
        try:
            # base64 补足 padding
            padded = encoded + "=" * (-len(encoded) % 4)
            return base64.urlsafe_b64decode(padded).decode("utf-8")
        except Exception:
            logger.debug("Bing URL 解码失败，使用原始链接: %s", ck_url)
    return ck_url


def _search_bing(query: str, top_k: int, timeout: float) -> list[dict[str, str]]:
    """必应搜索后端：抓取 cn.bing.com 搜索结果页并解析标题/摘要/链接。

    Returns:
        形如 [{"title": ..., "snippet": ..., "url": ...}] 的结果列表。
    """
    resp = requests.get(
        "https://cn.bing.com/search",
        params={"q": query, "count": top_k, "setlang": "zh-hans"},
        headers=_HEADERS,
        timeout=timeout,
    )
    resp.raise_for_status()
    text = resp.text

    # 反爬识别：正常结果页含 b_algo 结果块；验证/拦截页则直接报错
    if "b_algo" not in text and ("请输入验证码" in text or "检测到" in text):
        raise SearchError(
            "必应返回了安全验证页面（可能触发反爬），请稍后重试或切换 search_engine",
            tool_name="web_search",
        )

    results: list[dict[str, str]] = []
    # 按结果块切分（<li class="b_algo"> ... </li>）
    blocks = re.findall(r'<li class="b_algo".*?</li>', text, re.DOTALL)
    for block in blocks[:top_k]:
        # 标题 + 链接
        match = re.search(
            r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a></h2>', block, re.DOTALL
        )
        if not match:
            continue
        raw_url = html_lib.unescape(match.group(1))
        title = re.sub(r"<[^>]+>", "", match.group(2))
        title = html_lib.unescape(title).strip()
        if not title:
            continue
        # 摘要（结果块内第一个 <p>）
        p_match = re.search(r"<p[^>]*>(.*?)</p>", block, re.DOTALL)
        snippet = (
            html_lib.unescape(re.sub(r"<[^>]+>", "", p_match.group(1))).strip()
            if p_match
            else ""
        )
        results.append({
            "title": title,
            "snippet": snippet[:300],
            "url": _decode_bing_url(raw_url),
        })
    return results


def _search_wikipedia(query: str, top_k: int, timeout: float) -> list[dict[str, str]]:
    """维基百科搜索后端：搜索 + 取每条的摘要片段。

    Returns:
        形如 [{"title": ..., "snippet": ..., "url": ...}] 的结果列表。
    """
    api_url = "https://zh.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": top_k,
        "format": "json",
        "utf8": 1,
    }
    resp = requests.get(api_url, params=params, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    results: list[dict[str, str]] = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        # 清理维基返回的 HTML 片段标记
        snippet = (
            snippet.replace("<span class=\"searchmatch\">", "")
            .replace("</span>", "")
            .replace("&quot;", "\"")
        )
        results.append({
            "title": title,
            "snippet": snippet,
            "url": f"https://zh.wikipedia.org/wiki/{requests.utils.quote(title)}",
        })
    return results


def _search_duckduckgo(query: str, top_k: int, timeout: float) -> list[dict[str, str]]:
    """DuckDuckGo Instant Answer 后端：对常见实体类问题返回结构化卡片。

    Note: 该后端主要覆盖「实体/定义/事实类」问题，长尾问题可能返回空。
    """
    api_url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    resp = requests.get(api_url, params=params, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    results: list[dict[str, str]] = []
    abstract = data.get("AbstractText", "")
    if abstract:
        results.append({
            "title": data.get("Heading", query),
            "snippet": abstract[:500],
            "url": data.get("AbstractURL", ""),
        })
    # 补充 RelatedTopics
    for topic in data.get("RelatedTopics", [])[:top_k]:
        if isinstance(topic, dict) and topic.get("Text"):
            results.append({
                "title": topic.get("FirstURL", query),
                "snippet": topic["Text"][:300],
                "url": topic.get("FirstURL", ""),
            })
    return results


def _search_with_retry(query: str, engine: str, top_k: int, timeout: float) -> list[dict[str, str]]:
    """带重试的搜索执行（对瞬时网络故障做有限退避重试）。"""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            if engine == "bing":
                return _search_bing(query, top_k, timeout)
            if engine == "wikipedia":
                return _search_wikipedia(query, top_k, timeout)
            return _search_duckduckgo(query, top_k, timeout)
        except SearchError:
            raise  # 业务性失败（如反爬识别）不重试，直接上抛
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_exc = exc
            logger.warning("搜索第 %d 次尝试失败: %s", attempt + 1, exc)
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BASE_DELAY * (2**attempt))  # 指数退避
    raise SearchError(
        f"网络搜索失败（已重试 {_MAX_RETRIES} 次）: {last_exc}",
        tool_name="web_search",
        cause=last_exc,
    )


@tool
def web_search(query: str) -> str:
    """在互联网上搜索给定关键词，返回前若干条结果（标题 + 摘要 + 来源链接）。

    用于获取实时信息、事实查证、用户问题超出本地知识时补充外部资料。

    参数:
        query: 搜索关键词，建议使用简洁、聚焦的短语而非整句提问。

    返回:
        搜索结果的 JSON 字符串（含 title / snippet / url 字段）。
    """
    logger.info("web_search 开始: query=%r", query)

    # 参数校验
    query = (query or "").strip()
    if not query:
        raise SearchError("搜索关键词不能为空", tool_name="web_search")
    if len(query) > 100:
        raise SearchError("搜索关键词过长（超过 100 字符）", tool_name="web_search")

    settings = get_settings()
    try:
        results = _search_with_retry(
            query=query,
            engine=settings.search_engine,
            top_k=settings.search_top_k,
            timeout=settings.search_timeout,
        )
    except SearchError:
        logger.exception("web_search 失败: query=%r", query)
        raise

    if not results:
        logger.info("web_search 无结果: query=%r", query)
        return json.dumps(
            {"query": query, "results": [], "message": "未找到相关结果，建议换个关键词"},
            ensure_ascii=False,
        )

    logger.info("web_search 完成: query=%r, 命中 %d 条", query, len(results))
    return json.dumps({"query": query, "results": results}, ensure_ascii=False)
