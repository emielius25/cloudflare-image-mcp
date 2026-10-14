"""Entry point - runs the whole autonomous pipeline once per invocation.

Steps: select product -> build affiliate link -> carousel + video -> caption ->
ManyLinks block -> save manual-upload assets -> log & update state.
"""

from __future__ import annotations

import datetime as dt
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

from . import add_manylink, make_images, select_product, write_copy
from .build_link import build_affiliate_link, check_link_live, validate_affiliate_link
from .util import (Config, LLM, LOG, OUTPUT_DIR, POSTED_FILE, append_summary,
                   load_config, load_dotenv, read_json, setup_logging, shorten,
                   write_json)


def _prepare_output_dir(asin: str, when: dt.date) -> Path:
    out_dir = OUTPUT_DIR / f"{when.isoformat()}-{asin}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_once(cfg: Config, llm: LLM, posted: dict[str, Any]) -> dict[str, Any] | None:
    """Run one full post cycle. Returns the state entry, or ``None`` to skip."""
    product = select_product.select_product(cfg, llm, posted)
    if not product:
        LOG.warning("no product cleared the bar - skipping this run")
        return None

    asin = product["asin"]
    today = dt.datetime.now(dt.timezone.utc).date()
    out_dir = _prepare_output_dir(asin, today)
    LOG.info("output directory: %s", out_dir)

    try:
        # STEP 2 - affiliate link
        affiliate_url = build_affiliate_link(
            product.get("amazon_url"), asin, cfg.affiliate_tag, cfg.amazon_base)
        ok, reason = validate_affiliate_link(affiliate_url, cfg.affiliate_tag)
        if not ok:
            raise RuntimeError(f"affiliate link failed validation: {reason}")
        live, detail = check_link_live(affiliate_url)
        if not live:
            raise RuntimeError(f"affiliate link is not reachable: {detail}")
        LOG.info("affiliate link OK (%s)", detail)

        # STEP 3 - carousel + video
        carousel = make_images.make_carousel(product, cfg, llm, out_dir)

        # STEP 4 - caption
        caption = write_copy.build_caption(product, affiliate_url, cfg, llm)
        (out_dir / "caption.txt").write_text(caption, encoding="utf-8")

        # STEP 5 - ManyLinks bio page
        add_manylink.verify(cfg)
        title = shorten(product.get("title") or "Bedroom find", 60)
        description = (product.get("key_benefits") or ["A cozy bedroom upgrade"])[0]
        block = add_manylink.create_link_block(
            cfg, affiliate_url, title, shorten(description, 120))
        block_id = block.get("id")
        add_manylink.prune(cfg, cfg.max_manylink_blocks)

        # STEP 6 - assets for manual upload
        product_payload = dict(product)
        product_payload["affiliate_url"] = affiliate_url
        product_payload["manylinks_block_id"] = block_id
        product_payload["output_path"] = str(out_dir)
        product_payload["date"] = today.isoformat()
        product_payload["generations"] = carousel.get("generations")
        write_json(out_dir / "product.json", product_payload)
        _write_checklist(out_dir, product)

        # STEP 7 - state entry
        entry = {
            "date": today.isoformat(),
            "asin": asin,
            "title": product.get("title"),
            "affiliate_url": affiliate_url,
            "score": product.get("score"),
            "rationale": product.get("rationale"),
            "manylinks_block_id": block_id,
            "output_path": str(out_dir),
        }
        LOG.info("run complete: %s", asin)
        return entry
    except Exception:
        LOG.error("pipeline failed - discarding partial output in %s", out_dir)
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


def _write_checklist(out_dir: Path, product: dict[str, Any]) -> None:
    lines = [
        "# Upload checklist",
        "",
        f"**{shorten(product.get('title') or '', 70)}** (ASIN `{product.get('asin')}`)",
        "",
        "1. Upload `video.mp4` to TikTok",
        "2. Paste `caption.txt` in the description",
        "3. Enable the AI-generated content label",
        "4. Add a trending sound",
        "5. Confirm the link in bio works",
        "",
        "> This post is AI-generated. Label it as AI-generated in TikTok.",
        "",
        "Slides are in `slides/` for reference. `product.json` holds the full metadata.",
    ]
    (out_dir / "UPLOAD_CHECKLIST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_markdown(entries: list[dict[str, Any]], skipped: int,
                      failure: str | None) -> str:
    lines = ["## Amazon affiliate carousel run", ""]
    if failure:
        lines += [f"**Failed:** {failure}", ""]
    if not entries:
        lines += ["No product was posted this run."]
        if skipped:
            lines += [f"- Skipped: {skipped} (nothing scored >= 6.0)"]
    for entry in entries:
        lines += [
            f"### {shorten(entry.get('title') or entry['asin'], 70)}",
            f"- ASIN: `{entry['asin']}`",
            f"- Score: **{entry.get('score')}**",
            f"- Affiliate link: {entry.get('affiliate_url')}",
            f"- ManyLinks block: `{entry.get('manylinks_block_id')}`",
            f"- Output: `{entry.get('output_path')}`",
            "",
            "**Why this product**",
            "",
            entry.get("rationale") or "",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    load_dotenv()
    cfg = load_config()

    missing = cfg.missing_secrets()
    if missing:
        LOG.error("missing required secrets: %s", ", ".join(missing))
        append_summary(_summary_markdown([], 0, f"missing secrets: {', '.join(missing)}"))
        return 1

    llm = LLM(cfg)
    posted = read_json(POSTED_FILE, {"version": 1, "posted": []})
    posted.setdefault("posted", [])

    entries: list[dict[str, Any]] = []
    skipped = 0
    failure: str | None = None

    for attempt in range(cfg.posts_per_run):
        LOG.info("=== post %d/%d ===", attempt + 1, cfg.posts_per_run)
        try:
            entry = run_once(cfg, llm, posted)
        except Exception as exc:  # noqa: BLE001 - we want a clean non-zero exit
            failure = f"{type(exc).__name__}: {exc}"
            LOG.error("run failed: %s", failure)
            LOG.debug("%s", traceback.format_exc())
            break
        if entry is None:
            skipped += 1
            break
        entries.append(entry)
        posted["posted"].append(entry)

    if entries:
        write_json(POSTED_FILE, posted)

    append_summary(_summary_markdown(entries, skipped, failure))
    return 1 if failure else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
