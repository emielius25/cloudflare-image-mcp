# AGENT.md - Autonomous Amazon Affiliate Carousel Agent

This repository hosts an autonomous agent that, on every scheduled run, picks one
Amazon product in the configured niche, generates a TikTok photo-carousel video
for it, builds the tagged affiliate link, publishes that link to the ManyLinks
bio page and saves everything as ready-to-upload assets.

Uploading to TikTok is **manual** for now (see `output/<date>-<asin>/UPLOAD_CHECKLIST.md`).

## Pipeline

| Step | Module | What it does |
|------|--------|--------------|
| 1 | `src/select_product.py` | Smart product selection via the Axesso Amazon Data Service (RapidAPI) |
| 2 | `src/build_link.py` | Build + validate the tagged affiliate link |
| 3 | `src/make_images.py` | Generate 9:16 slides (Cloudflare Workers AI) + text overlays + MP4 |
| 4 | `src/write_copy.py` | Caption / description with disclosure, bullets and hashtags |
| 5 | `src/add_manylink.py` | Publish the link to the ManyLinks bio page |
| 6 | `src/main.py` | Orchestration, asset output, state, step summary |
| 7 | `data/posted.json` | Committed state: never repeat an ASIN within 90 days |

## Configuration (secrets / variables)

Never hardcode or log these. Locally copy `.env.example` to `.env`; in CI they
come from GitHub secrets and repository variables.

Secrets: `RAPIDAPI_KEY`, `MANYLINKS_API_KEY`, `CF_API_TOKEN`, `CF_ACCOUNT_ID`, `LLM_API_KEY`.

Variables: `AFFILIATE_TAG`, `NICHE`, `TARGET`, `POSTS_PER_RUN`.

Optional tuning: `LLM_BASE_URL`, `LLM_MODEL`, `LLM_VISION_MODEL`, `FFMPEG_BIN`,
`FFMPEG_PRESET`, `FONT_PATH`.

## Data sources (verified 2026-10-14)

Amazon data is read through the **Axesso Amazon Data Service** on RapidAPI - the
sanctioned commercial API. Amazon HTML is never scraped.

* Search: `GET /amz/amazon-search-by-keyword-asin`
* Detail: `GET /amz/amazon-lookup-product?url=...`

Images come from Cloudflare Workers AI `@cf/black-forest-labs/flux-1-schnell`
(Apache-2.0, free tier: 10,000 neurons/day). ffmpeg assembles the video.

## Selection logic

1. The LLM proposes 5-6 varied niche keywords (falls back to a static list).
2. Each keyword is searched (budget: <= 10 RapidAPI calls, `withCache=true`) and
   results are merged by ASIN.
3. Hard filters: rating >= 4.2, >= 500 reviews, price $10-$80, in stock, not
   posted in the last 90 days, no regulated/medical/adult items, no
   keyword-stuffed no-name brands.
4. Survivors are scored 0-10 on demand, deal depth, visual potential, impulse
   price and the quality of the listing's real `features[]`.
5. The top 3 get a product-detail lookup; the best is chosen and a 3-5 line
   "why this product / why not the runner-up" is written to the run log.
6. If the top score is below **6.0**, the run is skipped (exit 0) and says why.

## Affiliate link rules

* Base = `https://www.amazon.com` + `dpUrl`, or `/dp/<ASIN>` when missing.
* The tag is appended as a query parameter **at the end** (`?tag=` / `&tag=`).
* Any pre-existing `tag=` value is removed - exactly one remains.
* Links are never shortened, cloaked or redirected.
* Validated for host, single tag param and reachability before use.

## Assets & guardrails

* 6-8 slides, all 1080x1920, one shared art direction, no text drawn by the model.
* Overlay text is drawn locally (Pillow), <= 12 words, high contrast, kept out of
  the bottom ~20% and right ~15% (TikTok UI safe margins).
* One 1080x1920 H.264 30fps MP4, ~3-4s per slide, crossfades, 25-30s total.
* A vision model reviews every frame (legibility, artifacts, brand safety);
  failures regenerate up to 3 times, then the run aborts.
* Never re-host Amazon product images; never fake the branded product.
* Never log secrets; one product per run; disclosure in every caption.

## Amazon Associates compliance

Tagged links only, clear disclosure (`#ad As an Amazon Associate I earn from
qualifying purchases.`), no re-hosted Amazon images, no misleading or health
claims, and posts must be labelled AI-generated in TikTok.
