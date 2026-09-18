"""Credential-free public headline context for the dashboard."""

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List

import requests


FEEDS = (
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
)
POSITIVE = re.compile(r"\b(surge|rally|approval|inflow|bullish|breakout|record|gain)\b", re.I)
NEGATIVE = re.compile(r"\b(crash|hack|lawsuit|outflow|bearish|liquidation|ban|loss)\b", re.I)


def scan_headlines(timeout: float = 4.0, limit: int = 8) -> Dict:
    headlines: List[Dict] = []
    for feed in FEEDS:
        try:
            response = requests.get(feed, timeout=timeout, headers={"User-Agent": "INTRA-research/1.0"})
            response.raise_for_status()
            root = ET.fromstring(response.content)
            for item in root.findall(".//item")[:limit]:
                title = " ".join((item.findtext("title") or "").split())
                link = (item.findtext("link") or "").strip()
                if not title:
                    continue
                score = 1 if POSITIVE.search(title) else -1 if NEGATIVE.search(title) else 0
                headlines.append({"title": title[:240], "link": link, "bias": score})
        except (requests.RequestException, ET.ParseError):
            continue
    headlines = headlines[:limit]
    bias = sum(item["bias"] for item in headlines)
    direction = "CONSTRUCTIVE" if bias > 0 else "CAUTIOUS" if bias < 0 else "MIXED"
    return {
        "direction": direction,
        "headline_count": len(headlines),
        "headlines": headlines,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Headline context is supplemental and not a price guarantee.",
    }
