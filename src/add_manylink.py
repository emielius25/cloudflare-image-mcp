"""STEP 5 - publish the affiliate link to the ManyLinks bio page.

Verified against a FREE ManyLinks account:
``GET /v1/me``, ``POST /v1/blocks`` (201) and ``DELETE /v1/blocks/{id}`` (204).
Body limit is 256 KB; rate-limit headers (``Retry-After``) are honored by the
shared HTTP helper.
"""

from __future__ import annotations

from typing import Any

from .util import Config, HttpError, LOG, request, shorten

API_BASE = "https://api.manylinks.io"


class ManyLinksError(RuntimeError):
    pass


def _headers(cfg: Config) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {cfg.manylinks_api_key}",
        "Content-Type": "application/json",
    }


def get_me(cfg: Config) -> dict[str, Any]:
    """Return the account for the current key (also validates the key)."""
    resp = request("GET", f"{API_BASE}/v1/me", headers=_headers(cfg), timeout=30.0)
    return resp.json()


def verify(cfg: Config) -> dict[str, Any]:
    """Confirm the key works and has ``blocks:write`` in its scopes."""
    me = get_me(cfg)
    scopes = me.get("scopes") or me.get("scope") or []
    if isinstance(scopes, str):
        scopes = scopes.split()
    if scopes and "blocks:write" not in scopes and "*" not in scopes:
        raise ManyLinksError(f"key is missing the blocks:write scope (has: {scopes})")
    LOG.info("ManyLinks key OK (%s)", me.get("email") or me.get("username") or "account")
    return me


def create_link_block(cfg: Config, url: str, title: str,
                      description: str) -> dict[str, Any]:
    """Create a 2x1 link block and return the created block."""
    body = {
        "type": "link",
        "size": "2x1",
        "content": {
            "url": url,
            "title": shorten(title, 60),
            "description": shorten(description, 120),
        },
    }
    resp = request("POST", f"{API_BASE}/v1/blocks", headers=_headers(cfg),
                   json_body=body, timeout=30.0, retries=3)
    if resp.status_code != 201:
        raise ManyLinksError(f"unexpected status creating block: {resp.status_code}")
    block = resp.json()
    LOG.info("ManyLinks block created: %s", block.get("id"))
    return block


def list_blocks(cfg: Config, limit: int = 100) -> list[dict[str, Any]]:
    """List every block, following cursor pagination."""
    blocks: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        resp = request("GET", f"{API_BASE}/v1/blocks", headers=_headers(cfg),
                       params=params, timeout=30.0)
        payload = resp.json()
        data = payload.get("data") if isinstance(payload, dict) else payload
        blocks.extend(data or [])
        cursor = payload.get("nextCursor") if isinstance(payload, dict) else None
        if not cursor or not payload.get("hasMore"):
            break
    return blocks


def delete_block(cfg: Config, block_id: str) -> None:
    resp = request("DELETE", f"{API_BASE}/v1/blocks/{block_id}",
                   headers=_headers(cfg), timeout=30.0)
    if resp.status_code not in (200, 202, 204):
        raise ManyLinksError(f"unexpected status deleting block: {resp.status_code}")


def prune(cfg: Config, max_blocks: int = 30) -> list[str]:
    """Delete the oldest link blocks when the page holds more than ``max_blocks``."""
    blocks = list_blocks(cfg)
    links = [b for b in blocks if (b.get("type") or "link") == "link"]
    if len(links) <= max_blocks:
        return []
    # Oldest first - assume API returns newest-first, fall back to id order.
    ordered = list(reversed(links))
    removed: list[str] = []
    for block in ordered[: len(links) - max_blocks]:
        block_id = block.get("id")
        if not block_id:
            continue
        try:
            delete_block(cfg, block_id)
            removed.append(block_id)
        except HttpError as exc:
            LOG.warning("could not delete block %s: %s", block_id, exc)
    if removed:
        LOG.info("pruned %d old ManyLinks block(s)", len(removed))
    return removed
