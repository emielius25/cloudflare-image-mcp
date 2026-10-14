"""End-to-end verification for the affiliate carousel agent.

Runs offline: network calls (Amazon data, Cloudflare, ManyLinks, LLM) are
stubbed, but the real Pillow composition and ffmpeg assembly are exercised so we
can assert on the actual output files.

Run with:  python -m tests.verify_pipeline
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from src import add_manylink, main as agent_main, make_images  # noqa: E402
from src import select_product, write_copy  # noqa: E402
from src.build_link import (build_affiliate_link, build_amazon_url,  # noqa: E402
                            validate_affiliate_link)
from src.select_product import (hard_filter, normalise_candidate,  # noqa: E402
                                parse_price, parse_rating, parse_reviews,
                                parse_sales_volume, score_candidate)
from src.util import Config, LLM  # noqa: E402


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------

def test_build_link_rules() -> None:
    tag = "bedroomassoci-20"
    url = build_affiliate_link("/Cool-Lamp/dp/B0TEST1234/ref=sr_1_1", "B0TEST1234", tag)
    assert url == ("https://www.amazon.com/Cool-Lamp/dp/B0TEST1234/ref=sr_1_1"
                   "?tag=bedroomassoci-20"), url
    assert validate_affiliate_link(url, tag) == (True, "ok")

    assert build_affiliate_link(None, "B0TEST1234", tag) == (
        "https://www.amazon.com/dp/B0TEST1234?tag=bedroomassoci-20")

    messy = build_affiliate_link("/dp/B0TEST1234?th=1&tag=other-20", "B0TEST1234", tag)
    assert messy == "https://www.amazon.com/dp/B0TEST1234?th=1&tag=bedroomassoci-20", messy
    assert messy.count("tag=") == 1
    assert validate_affiliate_link(messy, tag) == (True, "ok")

    assert build_amazon_url("https://www.amazon.com/Foo/dp/B0X/ref=zz", "B0X") == (
        "https://www.amazon.com/Foo/dp/B0X/ref=zz")

    assert validate_affiliate_link("https://evil.com/dp/B0?tag=x", "x")[0] is False
    assert validate_affiliate_link("https://www.amazon.com/dp/B0", tag)[0] is False


def test_parsers() -> None:
    assert parse_rating("4.4 out of 5 stars") == 4.4
    assert parse_rating(4) == 4.0
    assert parse_reviews("1,234") == 1234
    assert parse_sales_volume("4K+ bought in past month") == 4000
    assert parse_sales_volume("1.2K+ bought") == 1200
    assert parse_sales_volume("900+ bought") == 900
    assert parse_price("$23.99") == 23.99
    assert parse_price("US$23.99 - $45.00") == 23.99


def test_filters_and_scoring() -> None:
    cfg = Config()
    good = normalise_candidate({
        "asin": "B0GOOD0001",
        "productDescription": "Cozy LED Bedside Lamp for Bedroom Decor with Dimmer",
        "price": "$29.99", "retailPrice": "$49.99",
        "productRating": "4.5 out of 5 stars", "countReview": "3,210",
        "salesVolume": "2K+ bought in past month", "dpUrl": "/x/dp/B0GOOD0001/ref=aa",
        "features": [
            "Warm dimmable light with three brightness levels for bedtime reading",
            "USB port charges your phone overnight without extra adapters",
            "Compact footprint fits any small nightstand",
        ],
    })
    ok, reason = hard_filter(good, set(), cfg)
    assert ok, reason
    score, breakdown = score_candidate(good, cfg)
    assert 6.0 <= score <= 10.0, (score, breakdown)

    assert hard_filter(dict(good, rating=3.1), set(), cfg)[0] is False
    assert hard_filter(dict(good, price=5.0), set(), cfg)[0] is False
    assert hard_filter(good, {"B0GOOD0001"}, cfg)[0] is False
    banned = normalise_candidate({
        "asin": "B0BAD00001",
        "productDescription": "CBD sleep supplement gummies for bedroom",
        "price": "$19.99", "productRating": "4.8 out of 5 stars",
        "countReview": "9000", "salesVolume": "5K+ bought",
    })
    assert hard_filter(banned, set(), cfg)[0] is False


def test_caption() -> None:
    cfg = Config()
    product = {
        "title": "Cozy LED Bedside Lamp",
        "key_benefits": ["Warm dimmable light for bedtime", "USB port on the base"],
        "rating": 4.5, "reviews": 3210, "price": 29.99, "retailPrice": 49.99,
        "hook_angle": "This tiny lamp made my bedroom feel like a hotel",
    }
    url = "https://www.amazon.com/dp/B0X?tag=bedroomassoci-20"
    caption = write_copy.build_caption(product, url, cfg, LLM(cfg))
    assert write_copy.DISCLOSURE in caption
    assert url in caption
    assert len(caption.splitlines()[0]) <= 90
    tag_line = caption.strip().splitlines()[-1]
    tags = tag_line.split()
    assert all(t.startswith("#") for t in tags), tags
    assert 5 <= len(tags) <= 8, tags


# ---------------------------------------------------------------------------
# Composition + video (real Pillow + ffmpeg)
# ---------------------------------------------------------------------------

def _fake_jpeg(width: int = 1024, height: int = 1024) -> bytes:
    base = Image.new("RGB", (16, 16))
    for y in range(16):
        for x in range(16):
            base.putpixel((x, y), (x * 16, 140, y * 16))
    img = base.resize((width, height), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_overlay_safe_margins() -> None:
    img = Image.open(io.BytesIO(_fake_jpeg()))
    out = make_images.draw_overlay(img, "This is a test headline for the bedroom")
    assert out.size == (1080, 1920)


def test_video_assembly(tmp_path: Path) -> None:
    slides = []
    for i in range(8):
        png = tmp_path / f"s{i}.png"
        make_images.compose_slide(_fake_jpeg(), f"Slide number {i + 1} headline",
                                  "Kicker", png)
        slides.append(png)
    video = make_images.assemble_video(slides, tmp_path / "video.mp4")
    assert video.exists() and video.stat().st_size > 10_000

    info = _probe(video)
    assert info["width"] == 1080, info
    assert info["height"] == 1920, info
    assert info["codec"] == "h264", info
    assert abs(float(info["fps"]) - 30) < 0.5, info
    assert 25 <= float(info["duration"]) <= 30, info


# ---------------------------------------------------------------------------
# Full pipeline (stubbed network)
# ---------------------------------------------------------------------------

def test_full_pipeline(tmp_path: Path) -> None:
    cfg = Config(affiliate_tag="bedroomassoci-20", posts_per_run=1)
    llm = LLM(cfg)  # disabled: no key -> deterministic fallbacks

    product = {
        "asin": "B0PIPE0001",
        "title": "Cozy LED Bedside Lamp for Bedroom Decor with Dimmer",
        "price": 29.99, "retailPrice": 49.99, "priceSaving": "-40%",
        "rating": 4.5, "reviews": 3210, "salesVolume": 2000,
        "amazon_url": "https://www.amazon.com/Cool-Lamp/dp/B0PIPE0001/ref=sr_1_1",
        "key_benefits": ["Warm dimmable light for bedtime reading",
                         "USB port charges your phone overnight"],
        "target_buyer": "Cozy bedroom renters", "hook_angle": "Hotel vibes for $30",
        "score": 8.4, "score_breakdown": {"demand": 3.0},
        "rationale": "Great demand, real benefits, impulse price.",
        "brand": "LumiCo", "features": ["Warm dimmable light for bedtime reading"],
    }

    out_root = tmp_path / "output"
    agent_main.OUTPUT_DIR = out_root
    agent_main.POSTED_FILE = tmp_path / "posted.json"
    agent_main.check_link_live = lambda url, timeout=30.0: (True, "HTTP 200")
    select_product.select_product = lambda cfg, llm, posted: product
    make_images.generate_image = lambda cfg, prompt, steps=4: _fake_jpeg()
    make_images.review_frame = lambda llm, png, product: (True, "ok")
    add_manylink.verify = lambda cfg: {"scopes": ["blocks:write"]}
    add_manylink.create_link_block = lambda cfg, url, title, desc: {"id": "blk_123"}
    add_manylink.prune = lambda cfg, max_blocks=30: []

    entry = agent_main.run_once(cfg, llm, {"posted": []})
    assert entry is not None
    assert entry["asin"] == "B0PIPE0001"
    assert entry["manylinks_block_id"] == "blk_123"
    assert "tag=bedroomassoci-20" in entry["affiliate_url"]

    out_dir = out_root / f"{entry['date']}-B0PIPE0001"
    assert (out_dir / "video.mp4").exists()
    assert len(list((out_dir / "slides").glob("*.png"))) == 8
    assert (out_dir / "caption.txt").exists()
    assert (out_dir / "product.json").exists()
    assert (out_dir / "UPLOAD_CHECKLIST.md").exists()

    saved = json.loads((out_dir / "product.json").read_text())
    assert saved["affiliate_url"].endswith("?tag=bedroomassoci-20")
    assert saved["manylinks_block_id"] == "blk_123"

    caption = (out_dir / "caption.txt").read_text()
    assert write_copy.DISCLOSURE in caption
    assert entry["affiliate_url"] in caption
    assert "AI-generated" in (out_dir / "UPLOAD_CHECKLIST.md").read_text()


# ---------------------------------------------------------------------------
# helpers / runner
# ---------------------------------------------------------------------------

def _ffprobe() -> str:
    override = os.environ.get("FFPROBE_BIN")
    if override and Path(override).exists():
        return override
    sibling = Path(make_images.ffmpeg_bin()).with_name("ffprobe")
    if sibling.exists():
        return str(sibling)
    found = shutil.which("ffprobe")
    if not found:
        raise FileNotFoundError("ffprobe not found")
    return found


def _probe(video: Path) -> dict[str, str]:
    cmd = [
        _ffprobe(), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name,r_frame_rate,duration",
        "-of", "json", str(video),
    ]
    data = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    stream = data["streams"][0]
    fps = stream.get("r_frame_rate", "0/1")
    if "/" in fps:
        num, den = fps.split("/")
        fps = str(float(num) / float(den or 1))
    return {
        "width": stream["width"], "height": stream["height"],
        "codec": stream["codec_name"], "fps": fps,
        "duration": stream.get("duration", "0"),
    }


def _run_all() -> int:
    import tempfile
    import traceback

    simple = [
        test_build_link_rules, test_parsers, test_filters_and_scoring,
        test_caption, test_overlay_safe_margins,
    ]
    failures = 0
    for test in simple:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {test.__name__}\n{traceback.format_exc()}")

    for test in (test_video_assembly, test_full_pipeline):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                test(Path(tmp))
            print(f"PASS {test.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {test.__name__}\n{traceback.format_exc()}")

    print(f"\n{'ALL TESTS PASSED' if not failures else f'{failures} TEST(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
