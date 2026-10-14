"""STEP 4 - the TikTok caption / description.

The description leads with the hook, carries the affiliate disclosure near the
top, lists three real benefit bullets, a CTA, the tagged affiliate link on its
own line and 5-8 hashtags. No health claims, no invented stats, no fake scarcity.
"""

from __future__ import annotations

from typing import Any

from .util import Config, HttpError, LLM, LLMError, LOG, shorten

DISCLOSURE = "#ad As an Amazon Associate I earn from qualifying purchases."

DEFAULT_HASHTAGS = [
    "#bedroomfinds", "#amazonfinds", "#homedecor", "#bedroomdecor",
    "#cozyhome", "#tiktokmademebuyit", "#amazonmusthaves",
]

HOOK_LIMIT = 90


def hashtags_for(cfg: Config, extra: list[str] | None = None) -> list[str]:
    """Return 5-8 niche + broad hashtags."""
    niche_tags = []
    for word in cfg.niche.replace("&", " ").replace(",", " ").split():
        word = "".join(ch for ch in word.lower() if ch.isalnum())
        if len(word) >= 4:
            niche_tags.append(f"#{word}")
    tags: list[str] = []
    for tag in (extra or []) + niche_tags + DEFAULT_HASHTAGS:
        tag = tag if tag.startswith("#") else f"#{tag}"
        if tag not in tags:
            tags.append(tag)
    return tags[:8]


def _fallback_hook(product: dict[str, Any]) -> str:
    hook = product.get("hook_angle") or "This bedroom find changed the whole vibe"
    return shorten(hook, HOOK_LIMIT)


def _fallback_bullets(product: dict[str, Any]) -> list[str]:
    benefits = [b for b in (product.get("key_benefits") or []) if b]
    if not benefits:
        benefits = ["Easy upgrade that instantly makes the room feel calmer"]
    return [shorten(b, 110) for b in benefits[:3]]


def build_caption(product: dict[str, Any], affiliate_url: str, cfg: Config,
                  llm: LLM) -> str:
    """Compose the ready-to-paste description."""
    hook = _fallback_hook(product)
    bullets = _fallback_bullets(product)
    hashtags = hashtags_for(cfg)
    cta = "Tap the link in bio for the current price."

    if llm.enabled:
        facts = (
            f"Product: {product.get('title')}\n"
            f"Benefits (real, from the listing): {bullets}\n"
            f"Rating: {product.get('rating')} ({product.get('reviews')} reviews)\n"
            f"Price: ${product.get('price')} (was ${product.get('retailPrice')})\n"
            f"Target buyer: {product.get('target_buyer')}"
        )
        instructions = (
            "Write a TikTok description for this product. Reply with strict JSON: "
            '{"hook": "<= 90 characters", "bullets": ["exactly 3 short benefit lines"], '
            '"cta": "one short call to action", "hashtags": ["5-8 lowercase hashtags"]}. '
            "Honest, friendly, conversational. Never invent stats or reviews, no health "
            "or medical claims, no fake scarcity. If price is mentioned add "
            "'price at time of posting'."
        )
        try:
            data = llm.chat_json("You write honest TikTok product descriptions.",
                                 facts + "\n\n" + instructions, temperature=0.7)
            if isinstance(data, dict):
                hook = shorten(str(data.get("hook") or hook), HOOK_LIMIT)
                new_bullets = [shorten(str(b), 110) for b in (data.get("bullets") or []) if str(b).strip()]
                if new_bullets:
                    bullets = new_bullets[:3]
                cta = str(data.get("cta") or cta)
                hashtags = hashtags_for(cfg, [str(h) for h in (data.get("hashtags") or [])])
        except (LLMError, HttpError, ValueError) as exc:
            LOG.warning("caption generation failed, using template: %s", exc)

    lines = [hook, "", DISCLOSURE, ""]
    lines += [f"• {b}" for b in bullets]
    lines += ["", cta, "", affiliate_url, "", " ".join(hashtags)]
    return "\n".join(lines).strip() + "\n"


def caption_checklist(product: dict[str, Any]) -> list[str]:
    """Human-facing reminders surfaced in the upload checklist."""
    return [
        "Upload video.mp4 to TikTok",
        "Paste caption.txt in the description",
        "Enable the AI-generated content label",
        "Add a trending sound",
        "Confirm the link in bio works",
    ]
