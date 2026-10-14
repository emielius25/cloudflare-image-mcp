"""Build and validate Amazon Associates affiliate links.

Rules enforced here (see the agent spec):

* The base URL is ``https://www.amazon.com`` + the ``dpUrl`` returned by the
  Axesso search endpoint, or ``https://www.amazon.com/dp/<ASIN>`` when the
  ``dpUrl`` is missing.
* The associates tag is appended as a query parameter at the *end* of the URL
  (``?tag=...`` if there is no query string yet, otherwise ``&tag=...``).
* Any pre-existing ``tag=`` parameter is removed so the final URL contains
  exactly one ``tag`` parameter.
* Links are never shortened, cloaked or redirected.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

AMAZON_HOSTS = ("amazon.com", "www.amazon.com")


def _normalise_dp_url(dp_url: str | None, asin: str, base: str = "https://www.amazon.com") -> str:
    """Turn a (possibly relative) ``dpUrl`` into an absolute Amazon URL."""
    base = base.rstrip("/")
    if not dp_url:
        return f"{base}/dp/{asin}"

    candidate = dp_url.strip()
    if candidate.startswith("http://") or candidate.startswith("https://"):
        parts = urlsplit(candidate)
        path = parts.path or f"/dp/{asin}"
        query = parts.query
        return urlunsplit(("https", "www.amazon.com", path, query, ""))

    if not candidate.startswith("/"):
        candidate = "/" + candidate
    return base + candidate


def _set_tag(url: str, tag: str) -> str:
    """Return ``url`` with exactly one ``tag=<tag>`` query parameter, last."""
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if k.lower() != "tag"]
    query.append(("tag", tag))
    return urlunsplit((parts.scheme or "https", parts.netloc, parts.path,
                       urlencode(query), parts.fragment))


def build_amazon_url(dp_url: str | None, asin: str,
                     base: str = "https://www.amazon.com") -> str:
    """Canonical **un-tagged** Amazon product URL.

    STEP 1 stores this as ``amazon_url``; STEP 2 attaches the associates tag.
    """
    return _normalise_dp_url(dp_url, asin, base)


def build_affiliate_link(
    dp_url: str | None,
    asin: str,
    tag: str,
    base: str = "https://www.amazon.com",
) -> str:
    """Build the final, tagged affiliate URL.

    >>> build_affiliate_link("/Some-Product/dp/B012345678/ref=sr_1_3", "B012345678", "mytag-20")
    'https://www.amazon.com/Some-Product/dp/B012345678/ref=sr_1_3?tag=mytag-20'
    >>> build_affiliate_link(None, "B012345678", "mytag-20")
    'https://www.amazon.com/dp/B012345678?tag=mytag-20'
    >>> build_affiliate_link("/dp/B012345678?th=1&tag=old-20", "B012345678", "new-20")
    'https://www.amazon.com/dp/B012345678?th=1&tag=new-20'
    """
    if not asin:
        raise ValueError("asin is required to build an affiliate link")
    if not tag:
        raise ValueError("affiliate tag is required to build an affiliate link")
    return _set_tag(_normalise_dp_url(dp_url, asin, base), tag)


def validate_affiliate_link(url: str, tag: str) -> tuple[bool, str]:
    """Structurally validate an affiliate URL.

    Returns ``(ok, reason)``. Checks the scheme, the host, that there is exactly
    one ``tag`` parameter and that it equals ``tag``.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return False, f"unsupported scheme: {parts.scheme!r}"
    host = (parts.hostname or "").lower()
    if host not in AMAZON_HOSTS:
        return False, f"host is not amazon.com: {host!r}"
    tags = [v for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() == "tag"]
    if len(tags) != 1:
        return False, f"expected exactly one tag parameter, found {len(tags)}"
    if tags[0] != tag:
        return False, f"tag mismatch: {tags[0]!r} != {tag!r}"
    if not parts.path.startswith("/"):
        return False, "path is not absolute"
    return True, "ok"


def check_link_live(url: str, timeout: float = 30.0) -> tuple[bool, str]:
    """Best-effort liveness probe. Amazon often blocks bots, so a 403/503 is
    treated as a soft pass (the URL is still well formed)."""
    try:
        from .util import request  # local import to keep pure helpers dependency-free

        resp = request("GET", url, timeout=timeout, retries=2,
                       retry_statuses=(429, 500, 502, 503, 504))
        return True, f"HTTP {resp.status_code}"
    except Exception as exc:  # noqa: BLE001 - probe is advisory only
        status = getattr(exc, "status", None)
        if status in (403, 503):
            return True, f"HTTP {status} (Amazon bot protection; URL accepted)"
        return False, str(exc)


def _demo() -> None:
    cases = [
        ("/Some-Product/dp/B012345678/ref=sr_1_3", "B012345678",
         "https://www.amazon.com/Some-Product/dp/B012345678/ref=sr_1_3?tag=bedroomassoci-20"),
        (None, "B0ABCDEFGH",
         "https://www.amazon.com/dp/B0ABCDEFGH?tag=bedroomassoci-20"),
        ("/dp/B0ABCDEFGH?th=1&tag=old-20", "B0ABCDEFGH",
         "https://www.amazon.com/dp/B0ABCDEFGH?th=1&tag=bedroomassoci-20"),
    ]
    for dp_url, asin, expected in cases:
        got = build_affiliate_link(dp_url, asin, "bedroomassoci-20")
        assert got == expected, f"{got!r} != {expected!r}"
        ok, reason = validate_affiliate_link(got, "bedroomassoci-20")
        assert ok, reason
    print("build_link demo OK")


if __name__ == "__main__":  # pragma: no cover
    _demo()
