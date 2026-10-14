"""STEP 1 - smart product selection.

Reads Amazon data through the Axesso Amazon Data Service (RapidAPI) - the
sanctioned commercial data source. Amazon HTML is never scraped.

The module exposes small, pure helpers (parsers, filters, scoring) plus a
:func:`select_product` orchestrator that returns the winning product as a dict,
or ``None`` when nothing clears the bar.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any

from .util import (Config, Guard, HttpError, LLM, LLMError, LOG, POSTED_FILE,
                   read_json, request, shorten)

# Keywords used when the LLM is unavailable.
FALLBACK_KEYWORDS = [
    "bedroom decor aesthetic",
    "cozy bedroom lighting",
    "bedside organizer",
    "sleep mask for side sleepers",
    "sunrise alarm clock",
    "bedroom wall decor",
]

# Terms that disqualify a product outright (regulated / medical / adult).
BANNED_TERMS = [
    "supplement", "vitamin", "medication", "medicine", "prescription",
    "cbd", "thc", "nicotine", "vape", "weight loss pill", "fat burner",
    "testosterone", "viagra", "adult", "sex toy", "vibrator", "condom",
    "medical device", "cure", "treats insomnia", "antidepressant",
]


# ---------------------------------------------------------------------------
# Parsers (pure)
# ---------------------------------------------------------------------------

def parse_rating(value: Any) -> float | None:
    """Parse ``"4.4 out of 5 stars"`` (or a plain number) into a float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else None


def parse_reviews(value: Any) -> int:
    """Parse ``"1,234"`` / ``1234`` into an int (0 when unknown)."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"(\d[\d,]*)", str(value))
    return int(match.group(1).replace(",", "")) if match else 0


def parse_sales_volume(value: Any) -> int:
    """Parse ``"4K+ bought in past month"`` into ``4000``."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*([KkMm]?)", text)
    if not match:
        return 0
    number = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix == "k":
        number *= 1_000
    elif suffix == "m":
        number *= 1_000_000
    return int(number)


def parse_price(value: Any) -> float | None:
    """Parse ``"$23.99"``, ``"US$23.99 - $45.00"`` into the first float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"(\d+(?:[\.,]\d{1,2})?)", str(value).replace(",", ""))
    return float(match.group(1)) if match else None


def parse_discount_percent(price: float | None, retail: float | None,
                           price_saving: Any = None) -> float:
    """Best-effort discount percentage from price/retailPrice/priceSaving."""
    if price_saving:
        match = re.search(r"(\d+(?:\.\d+)?)", str(price_saving))
        if match:
            return float(match.group(1))
    if price and retail and retail > price:
        return round((retail - price) / retail * 100, 1)
    return 0.0


# ---------------------------------------------------------------------------
# RapidAPI access
# ---------------------------------------------------------------------------

def _rapidapi_headers(cfg: Config) -> dict[str, str]:
    return {
        "x-rapidapi-key": cfg.rapidapi_key,
        "x-rapidapi-host": cfg.rapidapi_host,
    }


def search_products(cfg: Config, keyword: str, guard: Guard) -> list[dict[str, Any]]:
    """Call the Axesso keyword search endpoint and return raw candidates."""
    if not guard.spend():
        return []
    url = f"https://{cfg.rapidapi_host}/amz/amazon-search-by-keyword-asin"
    params = {
        "domainCode": "com",
        "keyword": keyword,
        "page": 1,
        "excludeSponsored": "true",
        "sortBy": "relevanceblender",
        "withCache": "true",
    }
    LOG.info("search: %r", keyword)
    resp = request("GET", url, headers=_rapidapi_headers(cfg), params=params,
                   timeout=60.0, retries=3)
    payload = resp.json()
    items = payload.get("searchProductDetails") or payload.get("productDetails") or []
    return [item for item in items if isinstance(item, dict)]


def lookup_product(cfg: Config, asin: str, guard: Guard) -> dict[str, Any] | None:
    """Call the product-detail endpoint for a single ASIN."""
    if not guard.spend():
        return None
    url = f"https://{cfg.rapidapi_host}/amz/amazon-lookup-product"
    LOG.info("lookup: %s", asin)
    try:
        # `requests` percent-encodes the value of `url` for us.
        resp = request("GET", url, headers=_rapidapi_headers(cfg),
                       params={"url": f"https://www.amazon.com/dp/{asin}"},
                       timeout=60.0, retries=3)
    except HttpError as exc:
        LOG.warning("lookup failed for %s: %s", asin, exc)
        return None
    payload = resp.json()
    return payload if isinstance(payload, dict) else None


# ---------------------------------------------------------------------------
# Keyword generation
# ---------------------------------------------------------------------------

def generate_keywords(cfg: Config, llm: LLM, count: int = 6) -> list[str]:
    """Ask the LLM for varied, purchase-intent keywords; fall back to a list."""
    if llm.enabled:
        try:
            data = llm.chat_json(
                "You are an e-commerce trend researcher. Reply with strict JSON only.",
                (
                    f"Generate {count} varied Amazon search keywords for the niche "
                    f"\"{cfg.niche}\". Cover problems, needs, season and current trends. "
                    "Each keyword must be 2-5 words, lowercase, no brand names, no "
                    "punctuation. Respond as {\"keywords\": [\"...\", ...]}."
                ),
                temperature=0.9,
            )
            raw = data.get("keywords") if isinstance(data, dict) else data
            keywords = [str(k).strip().lower() for k in (raw or []) if str(k).strip()]
            # de-duplicate, keep order, cap the number of API calls
            seen: list[str] = []
            for kw in keywords:
                if kw not in seen:
                    seen.append(kw)
            if seen:
                return seen[: cfg.max_api_keywords]
        except (LLMError, HttpError, ValueError) as exc:
            LOG.warning("keyword generation failed, using fallback: %s", exc)
    return FALLBACK_KEYWORDS[: cfg.max_api_keywords]


# ---------------------------------------------------------------------------
# Normalisation, filtering and scoring (pure)
# ---------------------------------------------------------------------------

def _first(raw: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = raw.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def normalise_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a raw Axesso search record onto our internal shape."""
    asin = str(_first(raw, "asin", "productAsin", default="") or "").strip()
    price = parse_price(_first(raw, "price", "buyboxPrice"))
    retail = parse_price(_first(raw, "retailPrice", "listPrice"))
    title = str(_first(raw, "productDescription", "title", "productTitle", default="") or "").strip()
    return {
        "asin": asin,
        "title": title,
        "price": price,
        "retailPrice": retail,
        "priceSaving": _first(raw, "priceSaving", default=None),
        "rating": parse_rating(_first(raw, "productRating", "rating")),
        "reviews": parse_reviews(_first(raw, "countReview", "reviews", "ratingsTotal")),
        "salesVolume": parse_sales_volume(_first(raw, "salesVolume", "boughtInPastMonth")),
        "dpUrl": _first(raw, "dpUrl", "url", default=None),
        "imgUrl": _first(raw, "imgUrl", "imageUrl", default=None),
        "prime": bool(_first(raw, "prime", default=False)),
        "deliveryMessage": _first(raw, "deliveryMessage", default=None),
        "warehouseAvailability": _first(raw, "warehouseAvailability", default=None),
        "brand": _first(raw, "manufacturer", "brand", default=None),
        "features": [str(f).strip() for f in (_first(raw, "features", default=[]) or [])],
        "soldBy": _first(raw, "soldBy", default=None),
        "fulfilledBy": _first(raw, "fulfilledBy", default=None),
        "raw": raw,
    }


def looks_like_keyword_stuffing(title: str) -> bool:
    """Cheap heuristic for spammy, keyword-stuffed, no-name listings."""
    if not title:
        return True
    words = title.split()
    if len(words) < 3:
        return True
    caps = [w for w in words if w.isupper() and len(w) > 2]
    if len(caps) >= 4:
        return True
    if len(set(w.lower() for w in words)) / max(len(words), 1) < 0.5:
        return True
    return False


def contains_banned_terms(text: str) -> bool:
    lowered = (text or "").lower()
    return any(term in lowered for term in BANNED_TERMS)


def recent_asins(posted: dict[str, Any], window_days: int,
                 now: dt.date | None = None) -> set[str]:
    """ASINs posted within the repeat window (default 90 days)."""
    now = now or dt.date.today()
    cutoff = now - dt.timedelta(days=window_days)
    recent: set[str] = set()
    for entry in (posted or {}).get("posted", []):
        asin = entry.get("asin")
        if not asin:
            continue
        try:
            when = dt.date.fromisoformat(str(entry.get("date", ""))[:10])
        except ValueError:
            recent.add(asin)
            continue
        if when >= cutoff:
            recent.add(asin)
    return recent


def hard_filter(cand: dict[str, Any], recent: set[str], cfg: Config) -> tuple[bool, str]:
    """Apply the non-negotiable filters. Returns ``(ok, reason)``."""
    if not cand.get("asin"):
        return False, "missing ASIN"
    if cand["asin"] in recent:
        return False, "posted within the last 90 days"
    title = cand.get("title") or ""
    if contains_banned_terms(title):
        return False, "regulated/medical/adult item"
    if looks_like_keyword_stuffing(title):
        return False, "keyword-stuffed / no-name brand"
    if cand.get("rating") is None or cand["rating"] < cfg.min_rating:
        return False, f"rating below {cfg.min_rating}"
    if cand.get("reviews", 0) < cfg.min_reviews:
        return False, f"fewer than {cfg.min_reviews} reviews"
    price = cand.get("price")
    if price is None or price < cfg.min_price or price > cfg.max_price:
        return False, f"price outside ${cfg.min_price:.0f}-${cfg.max_price:.0f}"
    availability = cand.get("warehouseAvailability")
    if availability and "in stock" not in str(availability).lower():
        return False, f"not in stock ({availability})"
    return True, "ok"


# Visual-potential vocabulary for lifestyle/scene scenes.
VISUAL_KEYWORDS = [
    "decor", "light", "lamp", "led", "clock", "organizer", "storage",
    "bedding", "sheet", "pillow", "blanket", "duvet", "curtain", "shelf",
    "mirror", "plant", "plush", "cozy", "aesthetic", "wall", "frame",
    "nightstand", "rug", "basket", "tray", "diffuser", "speaker",
]


def score_candidate(cand: dict[str, Any], cfg: Config) -> tuple[float, dict[str, float]]:
    """Score a candidate 0-10 across demand, deal, visuals, price and benefits."""
    breakdown: dict[str, float] = {}

    # Demand (max 3.0)
    sales = cand.get("salesVolume", 0) or 0
    reviews = cand.get("reviews", 0) or 0
    demand = 0.0
    if sales >= 4000:
        demand += 1.5
    elif sales >= 1000:
        demand += 1.1
    elif sales >= 300:
        demand += 0.7
    elif sales > 0:
        demand += 0.3
    if reviews >= 5000:
        demand += 1.5
    elif reviews >= 2000:
        demand += 1.2
    elif reviews >= 1000:
        demand += 0.8
    elif reviews >= cfg.min_reviews:
        demand += 0.4
    breakdown["demand"] = min(3.0, demand)

    # Deal (max 2.0)
    discount = parse_discount_percent(cand.get("price"), cand.get("retailPrice"),
                                      cand.get("priceSaving"))
    if discount >= 25:
        deal = 2.0
    elif discount >= 15:
        deal = 1.5
    elif discount >= 8:
        deal = 1.0
    elif discount > 0:
        deal = 0.5
    else:
        deal = 0.2
    breakdown["deal"] = deal

    # Visual potential (max 2.0)
    title = (cand.get("title") or "").lower()
    hits = sum(1 for word in VISUAL_KEYWORDS if word in title)
    breakdown["visual"] = min(2.0, 0.6 + hits * 0.5) if hits else 0.6

    # Impulse price (max 1.5)
    price = cand.get("price") or 0
    if 20 <= price <= 60:
        impulse = 1.5
    elif 15 <= price < 20 or 60 < price <= 70:
        impulse = 1.0
    elif price:
        impulse = 0.5
    else:
        impulse = 0.0
    breakdown["impulse"] = impulse

    # Real benefits (max 1.5)
    features = cand.get("features") or []
    usable = [f for f in features if 20 <= len(f) <= 240]
    benefit = 0.0
    if len(usable) >= 4:
        benefit = 1.5
    elif len(usable) == 3:
        benefit = 1.2
    elif len(usable) == 2:
        benefit = 0.8
    elif len(usable) == 1:
        benefit = 0.4
    breakdown["benefits"] = benefit

    total = round(min(10.0, sum(breakdown.values())), 2)
    return total, breakdown


# ---------------------------------------------------------------------------
# Detail merge + output shaping
# ---------------------------------------------------------------------------

def merge_detail(cand: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    """Overlay authoritative product-detail fields onto a search candidate."""
    d = normalise_candidate(detail)
    merged = dict(cand)
    for key in ("title", "price", "retailPrice", "priceSaving", "rating",
                "reviews", "warehouseAvailability", "brand", "soldBy",
                "fulfilledBy", "features"):
        value = d.get(key)
        if value not in (None, "", []):
            merged[key] = value
    return merged


def pick_key_benefits(features: list[str], limit: int = 5) -> list[str]:
    """Select short, concrete, believable benefit bullets from ``features``."""
    chosen: list[str] = []
    for raw in features:
        text = raw.strip().lstrip("•-*·▪● ").strip()
        text = re.sub(r"\s+", " ", text)
        if not (20 <= len(text) <= 180):
            continue
        if contains_banned_terms(text):  # never surface regulated claims
            continue
        if text in chosen:
            continue
        chosen.append(text)
        if len(chosen) >= limit:
            break
    return chosen


def _fallback_copy(cand: dict[str, Any], cfg: Config) -> dict[str, Any]:
    title = shorten(cand.get("title") or "this bedroom find", 60)
    return {
        "target_buyer": f"People furnishing or upgrading a cozy {cfg.niche} space.",
        "hook_angle": f"Small {cfg.niche} upgrade that makes the room feel calmer.",
        "key_benefits": [],
        "rationale": (
            f"{title} clears every hard filter (rating {cand.get('rating')}, "
            f"{cand.get('reviews')} reviews, strong sales volume) and sits in the "
            "impulse price band, so it is easy to feature on a short video."
        ),
    }


def _llm_copy(cand: dict[str, Any], cfg: Config, llm: LLM,
              runner_up: dict[str, Any] | None) -> dict[str, Any]:
    """Ask the LLM for the buyer, hook, benefits and the 'why' narrative."""
    fallback = _fallback_copy(cand, cfg)
    if not llm.enabled:
        return fallback

    runner_up_txt = "none"
    if runner_up:
        runner_up_txt = (f"{shorten(runner_up.get('title', ''), 70)} "
                         f"(score {runner_up.get('score')}, "
                         f"{runner_up.get('reviews')} reviews)")
    facts = (
        f"Product: {cand.get('title')}\n"
        f"Price: ${cand.get('price')} (was ${cand.get('retailPrice')}, "
        f"saving {cand.get('priceSaving')})\n"
        f"Rating: {cand.get('rating')} from {cand.get('reviews')} reviews\n"
        f"Sales: {cand.get('salesVolume')} bought recently\n"
        f"Features: {json.dumps(cand.get('features', [])[:6])}\n"
        f"Runner-up: {runner_up_txt}\n"
        f"Niche: {cfg.niche}"
    )
    instructions = (
        "Using ONLY the facts above, reply with strict JSON: "
        '{"target_buyer": "...", "hook_angle": "...", '
        '"key_benefits": ["3-5 concrete benefit bullets derived from Features"], '
        '"rationale": "3-5 lines: why this product, and one line why not the '
        'runner-up", "why_not_runner_up": "one line"}'
        " Never invent statistics, health claims or fake reviews."
    )
    try:
        data = llm.chat_json("You write honest, high-converting product content.",
                             facts + "\n\n" + instructions, temperature=0.6)
    except (LLMError, HttpError, ValueError) as exc:
        LOG.warning("LLM copy generation failed, using fallback: %s", exc)
        return fallback

    if not isinstance(data, dict):
        return fallback
    benefits = [str(b).strip() for b in (data.get("key_benefits") or []) if str(b).strip()]
    return {
        "target_buyer": str(data.get("target_buyer") or fallback["target_buyer"]),
        "hook_angle": str(data.get("hook_angle") or fallback["hook_angle"]),
        "key_benefits": benefits,
        "rationale": str(data.get("rationale") or fallback["rationale"]),
        "why_not_runner_up": str(data.get("why_not_runner_up") or ""),
    }


def to_output(cand: dict[str, Any], cfg: Config, copy: dict[str, Any]) -> dict[str, Any]:
    """Shape the STEP 1 JSON payload (the tag is added in STEP 2)."""
    from .build_link import build_amazon_url

    amazon_url = build_amazon_url(cand.get("dpUrl"), cand["asin"], cfg.amazon_base)

    benefits = copy.get("key_benefits") or pick_key_benefits(cand.get("features", []))
    if not benefits:
        benefits = [shorten(cand.get("title") or "Bedroom upgrade", 80)]

    return {
        "asin": cand["asin"],
        "title": cand.get("title"),
        "price": cand.get("price"),
        "retailPrice": cand.get("retailPrice"),
        "priceSaving": cand.get("priceSaving"),
        "rating": cand.get("rating"),
        "reviews": cand.get("reviews"),
        "salesVolume": cand.get("salesVolume"),
        "amazon_url": amazon_url,
        "key_benefits": benefits[:5],
        "target_buyer": copy.get("target_buyer"),
        "hook_angle": copy.get("hook_angle"),
        "score": cand.get("score"),
        "score_breakdown": cand.get("score_breakdown"),
        "rationale": copy.get("rationale"),
        "brand": cand.get("brand"),
        "features": cand.get("features", []),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def gather_candidates(cfg: Config, keywords: list[str], guard: Guard) -> dict[str, dict[str, Any]]:
    """Search every keyword (respecting budget) and merge results by ASIN."""
    candidates: dict[str, dict[str, Any]] = {}
    for keyword in keywords:
        # Keep at least 3 calls in reserve for the product-detail lookups.
        if guard.remaining < 3:
            LOG.info("stopping search early to reserve lookup budget")
            break
        try:
            raw_items = search_products(cfg, keyword, guard)
        except HttpError as exc:
            LOG.warning("search failed for %r: %s", keyword, exc)
            continue
        for raw in raw_items:
            cand = normalise_candidate(raw)
            asin = cand.get("asin")
            if asin and asin not in candidates:
                candidates[asin] = cand
    LOG.info("collected %d unique candidates from %d keywords",
             len(candidates), len(keywords))
    return candidates


def select_product(cfg: Config, llm: LLM,
                   posted: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return the winning product payload, or ``None`` when the run should skip."""
    posted = posted if posted is not None else read_json(POSTED_FILE, {"posted": []})
    guard = Guard("rapidapi", cfg.max_search_calls)

    keywords = generate_keywords(cfg, llm)
    LOG.info("keywords: %s", ", ".join(keywords))
    candidates = gather_candidates(cfg, keywords, guard)

    recent = recent_asins(posted, cfg.repeat_window_days)
    survivors: list[dict[str, Any]] = []
    for cand in candidates.values():
        ok, reason = hard_filter(cand, recent, cfg)
        if ok:
            survivors.append(cand)
        else:
            LOG.debug("rejected %s: %s", cand.get("asin"), reason)

    if not survivors:
        LOG.warning("no candidates passed the hard filters")
        return None

    for cand in survivors:
        cand["score"], cand["score_breakdown"] = score_candidate(cand, cfg)
    survivors.sort(key=lambda c: c["score"], reverse=True)

    top = survivors[:3]
    for cand in top:
        detail = lookup_product(cfg, cand["asin"], guard)
        if not detail:
            continue
        merged = merge_detail(cand, detail)
        merged["score"], merged["score_breakdown"] = score_candidate(merged, cfg)
        cand.clear()
        cand.update(merged)

    top.sort(key=lambda c: c["score"], reverse=True)
    best = top[0]
    runner_up = top[1] if len(top) > 1 else None

    if best["score"] < cfg.min_score:
        LOG.warning("best score %.2f is below the %.1f threshold - skipping run",
                    best["score"], cfg.min_score)
        return None

    copy = _llm_copy(best, cfg, llm, runner_up)
    output = to_output(best, cfg, copy)

    LOG.info("selected %s (%s) score=%.2f", output["asin"],
             shorten(output.get("title") or "", 60), best["score"])
    LOG.info("why this product:\n%s", copy.get("rationale"))
    if copy.get("why_not_runner_up"):
        LOG.info("why not the runner-up: %s", copy["why_not_runner_up"])
    LOG.info("score breakdown: %s", best.get("score_breakdown"))
    return output
