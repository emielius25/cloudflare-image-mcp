"""STEP 3 - carousel slides and the vertical video.

Generates 9:16 lifestyle scenes with Cloudflare Workers AI
(``@cf/black-forest-labs/flux-1-schnell``), adds a text overlay drawn locally
with Pillow, then stitches the slides into a 1080x1920 H.264 video with ffmpeg.

The model never draws text and never fakes the branded product - all copy is
composited by us and all scenes are generic lifestyle imagery.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .util import (ASSETS_DIR, CF_FLUX_MODEL, Config, Guard, HttpError, LLM,
                   LLMError, LOG, parse_json_loose, request, shorten)

WIDTH, HEIGHT = 1080, 1920
FPS = 30
SECONDS_PER_SLIDE = 4.0
CROSSFADE = 0.5

# Art direction shared by every slide: one palette, one mood, no text.
ART_DIRECTION = (
    "warm cozy bedroom, soft golden evening lamp light, photoreal editorial "
    "photography, shallow depth of field, 35mm, muted warm beige and amber "
    "palette, calm and inviting mood, high detail, no text, no logos, "
    "no watermark, no people's faces"
)

# Slide plan: (key, label, generic scene seed).
SLIDE_PLAN: list[tuple[str, str, str]] = [
    ("hook", "Hook", "a cozy bedroom corner at dusk, softly rumpled linen bedding, one warm lamp glowing"),
    ("problem", "Problem", "a dim, cluttered bedroom at night, tangled cables on a wooden nightstand, moody shadows"),
    ("solution", "Solution", "a serene tidy bedroom with warm ambient lighting, calm inviting atmosphere"),
    ("benefit1", "Benefit 1", "a close-up of a tidy nightstand scene with a warm glowing light and a small plant"),
    ("benefit2", "Benefit 2", "a quiet bedroom reading nook at night, warm lamp glow, cozy textures"),
    ("proof", "Social proof", "a cozy bedroom with soft fairy lights and greenery, evening ambience"),
    ("value", "Value", "a bright airy bedroom in morning light, minimal warm decor, calm mood"),
    ("cta", "CTA", "a person relaxing on a made bed in a warm, softly lit bedroom, peaceful mood"),
]

FALLBACK_OVERLAYS = {
    "hook": "The bedroom upgrade nobody talks about",
    "problem": "Your bedroom could feel so much calmer",
    "solution": "Here's the simple fix",
    "cta": "Link in bio",
}

MAX_OVERLAY_WORDS = 12


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

FONT_CANDIDATES_BOLD = [
    ASSETS_DIR / "fonts" / "NotoSans-Bold.ttf",
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
]
FONT_CANDIDATES_REGULAR = [
    ASSETS_DIR / "fonts" / "NotoSans-Regular.ttf",
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
]


def resolve_font(bold: bool = True) -> Path:
    """Find a usable TTF, preferring the bundled Noto Sans."""
    env_key = "FONT_BOLD_PATH" if bold else "FONT_REGULAR_PATH"
    override = os.environ.get(env_key) or os.environ.get("FONT_PATH")
    if override and Path(override).exists():
        return Path(override)
    for candidate in (FONT_CANDIDATES_BOLD if bold else FONT_CANDIDATES_REGULAR):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "no usable TrueType font found; set FONT_PATH or add assets/fonts/*.ttf"
    )


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def clamp_words(text: str, limit: int = MAX_OVERLAY_WORDS) -> str:
    """Keep overlay copy short: at most ``limit`` words on a single line."""
    words = (text or "").split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(",.;:") + "…"


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Greedy word wrap using real glyph widths."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if font.getlength(trial) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# Cloudflare Workers AI image generation
# ---------------------------------------------------------------------------

class ImageGenError(RuntimeError):
    pass


class NeuronsExhausted(ImageGenError):
    """Raised when the daily free neuron allowance is used up."""


def _cf_url(cfg: Config) -> str:
    return (f"https://api.cloudflare.com/client/v4/accounts/{cfg.cf_account_id}"
            f"/ai/run/{CF_FLUX_MODEL}")


def generate_image(cfg: Config, prompt: str, steps: int = 4) -> bytes:
    """Generate one image and return the decoded JPEG bytes."""
    prompt = prompt.strip()[:2048]
    resp = request(
        "POST",
        _cf_url(cfg),
        headers={
            "Authorization": f"Bearer {cfg.cf_api_token}",
            "Content-Type": "application/json",
        },
        json_body={"prompt": prompt, "steps": steps},
        timeout=120.0,
        retries=4,
    )
    try:
        payload = resp.json()
    except ValueError as exc:
        raise ImageGenError(f"non-JSON response from Workers AI: {exc}") from exc

    if not payload.get("success", True):
        message = json.dumps(payload.get("errors") or payload)[:300]
        lowered = message.lower()
        if "neuron" in lowered or "quota" in lowered or "limit" in lowered:
            raise NeuronsExhausted(message)
        raise ImageGenError(message)

    result = payload.get("result") or {}
    b64 = result.get("image")
    if not b64:
        raise ImageGenError("Workers AI response did not contain result.image")
    try:
        return base64.b64decode(b64)
    except (ValueError, TypeError) as exc:
        raise ImageGenError(f"could not decode image payload: {exc}") from exc


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def cover_resize(img: Image.Image, size: tuple[int, int] = (WIDTH, HEIGHT)) -> Image.Image:
    """Cover-fit ``img`` into ``size`` (crop, never stretch)."""
    target_w, target_h = size
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_size = (max(1, math.ceil(src_w * scale)), max(1, math.ceil(src_h * scale)))
    resized = img.resize(new_size, Image.LANCZOS)
    left = (new_size[0] - target_w) // 2
    top = (new_size[1] - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def _gradient_scrim(size: tuple[int, int], top_alpha: int = 0,
                    bottom_alpha: int = 215) -> Image.Image:
    """Vertical dark gradient used to guarantee text contrast."""
    width, height = size
    strip = Image.new("L", (1, height))
    for y in range(height):
        ratio = y / max(height - 1, 1)
        strip.putpixel((0, y), int(top_alpha + (bottom_alpha - top_alpha) * (ratio ** 1.6)))
    alpha = strip.resize((width, height))
    scrim = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    scrim.putalpha(alpha)
    return scrim


def draw_overlay(img: Image.Image, headline: str, kicker: str | None = None) -> Image.Image:
    """Draw the slide copy with a scrim, respecting TikTok safe margins.

    Text is confined to the top ~80% and left ~85% of the frame, so nothing is
    hidden behind the TikTok UI (bottom bar and right action rail).
    """
    canvas = cover_resize(img.convert("RGB"))
    overlay = canvas.convert("RGBA")

    scrim = _gradient_scrim(canvas.size)
    overlay = Image.alpha_composite(overlay, scrim)

    left_margin = int(WIDTH * 0.06)
    right_margin = int(WIDTH * 0.15)
    text_width = WIDTH - left_margin - right_margin
    bottom_limit = int(HEIGHT * 0.80)

    bold_font = ImageFont.truetype(str(resolve_font(bold=True)), 84)
    kicker_font = ImageFont.truetype(str(resolve_font(bold=False)), 44)

    headline = clamp_words(headline)
    lines = wrap_text(headline, bold_font, text_width)
    line_height = int(84 * 1.18)
    block_height = line_height * len(lines)

    kicker_txt = clamp_words(kicker, 6) if kicker else ""
    kicker_h = 0
    if kicker_txt:
        kicker_h = 72

    y = bottom_limit - block_height - kicker_h - 24
    y = max(int(HEIGHT * 0.30), y)

    draw = ImageDraw.Draw(overlay)
    if kicker_txt:
        draw.text((left_margin, y), kicker_txt.upper(), font=kicker_font,
                  fill=(255, 214, 140, 255))
        y += kicker_h

    for line in lines:
        draw.text((left_margin + 3, y + 3), line, font=bold_font, fill=(0, 0, 0, 170))
        draw.text((left_margin, y), line, font=bold_font, fill=(255, 255, 255, 255))
        y += line_height

    return overlay.convert("RGB")


def compose_slide(jpeg_bytes: bytes, headline: str, kicker: str | None,
                  out_path: Path, quality: int = 92) -> Path:
    """Decode a generated image, add the overlay and save the finished PNG."""
    with Image.open(io.BytesIO(jpeg_bytes)) as img:
        final = draw_overlay(img, headline, kicker)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(out_path, format="PNG")
    return out_path


# ---------------------------------------------------------------------------
# Overlay copy
# ---------------------------------------------------------------------------

def _fallback_overlays(product: dict[str, Any]) -> dict[str, dict[str, str]]:
    benefits = list(product.get("key_benefits") or [])
    b1 = benefits[0] if benefits else "Designed to make the room feel calmer"
    b2 = benefits[1] if len(benefits) > 1 else "Simple, tidy upgrade you'll actually use"
    rating = product.get("rating")
    reviews = product.get("reviews")
    proof = (f"{rating}★ from {int(reviews):,} reviews"
             if rating and reviews else "Loved by thousands of shoppers")
    saving = product.get("priceSaving")
    value = f"Save {saving} - price at time of posting" if saving else "Check the current price"
    hook = product.get("hook_angle") or "The bedroom upgrade nobody talks about"
    title = shorten(product.get("title") or "This bedroom find", 60)
    return {
        "hook": {"headline": hook, "kicker": "Bedroom find"},
        "problem": {"headline": "Your bedroom could feel so much calmer", "kicker": "The problem"},
        "solution": {"headline": f"Meet {title}", "kicker": "The fix"},
        "benefit1": {"headline": b1, "kicker": "Why you'll love it"},
        "benefit2": {"headline": b2, "kicker": "Also great"},
        "proof": {"headline": proof, "kicker": "Real reviews"},
        "value": {"headline": value, "kicker": "The deal"},
        "cta": {"headline": "Link in bio", "kicker": "Shop it"},
    }


def build_overlay_copy(product: dict[str, Any], cfg: Config,
                       llm: LLM) -> dict[str, dict[str, str]]:
    """Produce <=12-word overlay copy for each slide (LLM, else templates)."""
    fallback = _fallback_overlays(product)
    if not llm.enabled:
        return fallback

    facts = (
        f"Title: {product.get('title')}\n"
        f"Hook angle: {product.get('hook_angle')}\n"
        f"Benefits: {json.dumps(product.get('key_benefits') or [])}\n"
        f"Rating: {product.get('rating')} ({product.get('reviews')} reviews)\n"
        f"Saving: {product.get('priceSaving')}\n"
        f"Target buyer: {product.get('target_buyer')}"
    )
    instructions = (
        "Write short TikTok carousel overlay text. Reply with strict JSON with keys "
        "hook, problem, solution, benefit1, benefit2, proof, value, cta. "
        "Each value must be <= 12 words, punchy, honest and free of emojis. "
        "Use ONLY the given rating/review numbers for social proof - never invent "
        "stats. Do not mention health or medical benefits."
    )
    try:
        data = llm.chat_json("You write scroll-stopping but honest short-form copy.",
                             facts + "\n\n" + instructions, temperature=0.7)
    except (LLMError, HttpError, ValueError) as exc:
        LOG.warning("overlay copy generation failed, using templates: %s", exc)
        return fallback

    if not isinstance(data, dict):
        return fallback
    result: dict[str, dict[str, str]] = {}
    for key, _, _ in SLIDE_PLAN:
        value = data.get(key)
        headline = value.get("headline") if isinstance(value, dict) else value
        headline = clamp_words(str(headline)) if headline else fallback[key]["headline"]
        result[key] = {"headline": headline, "kicker": fallback[key]["kicker"]}
    return result


def scene_prompt(seed: str, cfg: Config) -> str:
    """Combine a generic scene seed with the shared art direction."""
    return f"{seed}, {ART_DIRECTION}"


# ---------------------------------------------------------------------------
# Video assembly (ffmpeg)
# ---------------------------------------------------------------------------

def ffmpeg_bin() -> str:
    override = os.environ.get("FFMPEG_BIN")
    if override and Path(override).exists():
        return override
    found = shutil.which("ffmpeg")
    if not found:
        raise FileNotFoundError("ffmpeg not found; set FFMPEG_BIN or install ffmpeg")
    return found


def _preset() -> str:
    """x264 preset (overridable for faster local/CI test runs)."""
    return os.environ.get("FFMPEG_PRESET", "medium")


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd[:4])}...\n"
            f"{proc.stderr[-800:]}"
        )


def _zoom_filter(index: int, frames: int) -> str:
    """Return a subtle Ken Burns zoompan filter for slide ``index``."""
    mode = index % 3
    if mode == 0:
        z = "min(1+0.0010*on,1.12)"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif mode == 1:
        z = "max(1.12-0.0010*on,1.0)"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    else:
        z = "1.12"
        x = f"(iw-iw/zoom)*on/{max(frames - 1, 1)}"
        y = "ih/2-(ih/zoom/2)"
    return (f"scale={WIDTH*2}:{HEIGHT*2},"
            f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
            "format=yuv420p")


def make_clip(png: Path, out: Path, index: int,
              duration: float = SECONDS_PER_SLIDE) -> Path:
    frames = max(2, int(duration * FPS))
    _run([
        ffmpeg_bin(), "-y", "-i", str(png),
        "-vf", _zoom_filter(index, frames),
        "-frames:v", str(frames),
        "-c:v", "libx264", "-preset", _preset(), "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        str(out),
    ])
    return out


def assemble_video(slides: list[Path], out_path: Path,
                   duration: float = SECONDS_PER_SLIDE,
                   crossfade: float = CROSSFADE) -> Path:
    """Stitch slides into one 1080x1920 H.264 video with crossfades."""
    if len(slides) < 2:
        raise ValueError("need at least two slides to assemble a video")

    with tempfile.TemporaryDirectory() as tmp:
        clips = []
        for i, slide in enumerate(slides):
            clip = Path(tmp) / f"clip_{i:02d}.mp4"
            make_clip(slide, clip, i, duration)
            clips.append(clip)

        cmd = [ffmpeg_bin(), "-y"]
        for clip in clips:
            cmd += ["-i", str(clip)]

        step = duration - crossfade
        parts = []
        last = "[0:v]"
        for i in range(1, len(clips)):
            offset = round(i * step, 3)
            out_label = f"[v{i}]"
            parts.append(
                f"{last}[{i}:v]xfade=transition=fade:duration={crossfade}:"
                f"offset={offset}{out_label}"
            )
            last = out_label
        filter_complex = ";".join(parts)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd += [
            "-filter_complex", filter_complex,
            "-map", last,
            "-c:v", "libx264", "-preset", _preset(), "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", str(FPS),
            "-movflags", "+faststart",
            str(out_path),
        ]
        _run(cmd)
    return out_path


# ---------------------------------------------------------------------------
# Vision QA
# ---------------------------------------------------------------------------

def frame_data_url(png: Path, max_width: int = 540) -> str:
    """Encode a frame as a small JPEG data URL for the vision reviewer."""
    with Image.open(png) as img:
        frame = img.convert("RGB")
        frame.thumbnail((max_width, max_width * 2))
        buffer = io.BytesIO()
        frame.save(buffer, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def review_frame(llm: LLM, png: Path, product: dict[str, Any]) -> tuple[bool, str]:
    """Ask the vision model whether a frame is safe and legible.

    Returns ``(ok, reason)``. When the LLM is unavailable we optimistically pass
    (the pipeline still guards on structure/size).
    """
    if not llm.enabled:
        return True, "vision review skipped (no LLM)"
    prompt = (
        "Review this single TikTok carousel frame. The overlaid text must be fully "
        "legible, high contrast and not cropped; there must be no garbled letters, "
        "no bad anatomy or extra fingers, no watermark and nothing unsafe, sexual or "
        "misleading. Reply with strict JSON: "
        '{"legible": bool, "artifacts": bool, "brand_safe": bool, "ok": bool, '
        '"issues": ["short reason", ...]}'
    )
    try:
        text = llm.review_image(frame_data_url(png), prompt)
        data = parse_json_loose(text)
    except (LLMError, HttpError, ValueError) as exc:
        LOG.warning("vision review failed (%s) - accepting frame", exc)
        return True, "review unavailable"
    if not isinstance(data, dict):
        return True, "review unparsable"
    ok = bool(data.get("ok")) and bool(data.get("legible", True)) \
        and bool(data.get("brand_safe", True)) and not bool(data.get("artifacts", False))
    reason = "; ".join(str(i) for i in (data.get("issues") or []))[:200] or "ok"
    return ok, reason


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def make_carousel(product: dict[str, Any], cfg: Config, llm: LLM,
                  out_dir: Path) -> dict[str, Any]:
    """Generate slides, QA them and assemble the vertical video."""
    out_dir.mkdir(parents=True, exist_ok=True)
    slides_dir = out_dir / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    overlays = build_overlay_copy(product, cfg, llm)
    guard = Guard("image-generations", cfg.max_image_generations)
    frames: list[Path] = []
    total_generations = 0

    for index, (key, label, seed) in enumerate(SLIDE_PLAN):
        text = overlays[key]
        png = slides_dir / f"{index + 1:02d}-{key}.png"
        prompt = scene_prompt(seed, cfg)
        attempts = 0
        while True:
            attempts += 1
            if total_generations >= cfg.max_image_generations + 3 * len(SLIDE_PLAN):
                raise ImageGenError("exceeded the hard generation ceiling")
            jpeg = generate_image(cfg, prompt)
            total_generations += 1
            if total_generations > cfg.max_image_generations:
                LOG.warning("generation budget (%d) exceeded at %d (QA retries)",
                            cfg.max_image_generations, total_generations)
            compose_slide(jpeg, text["headline"], text["kicker"], png)
            ok, reason = review_frame(llm, png, product)
            if ok:
                LOG.info("slide %d/%d (%s) ok after %d attempt(s)",
                         index + 1, len(SLIDE_PLAN), label, attempts)
                break
            if attempts >= 3:
                raise ImageGenError(
                    f"slide {index + 1} ({key}) failed QA after {attempts} attempts: {reason}")
            LOG.warning("slide %d (%s) failed QA (attempt %d): %s - regenerating",
                        index + 1, key, attempts, reason)
        frames.append(png)

    guard.used = total_generations
    video = assemble_video(frames, out_dir / "video.mp4")
    LOG.info("assembled %s (%d slides, %d generations)",
             video, len(frames), total_generations)
    return {
        "video": video,
        "slides": frames,
        "overlays": overlays,
        "generations": total_generations,
    }
