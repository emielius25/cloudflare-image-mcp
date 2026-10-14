"""Shared helpers for the Amazon affiliate carousel agent.

This module centralises configuration, structured logging, HTTP helpers with
retry/backoff, a small OpenAI-compatible LLM client and JSON file helpers.

Nothing in here ever logs (or returns) secret values.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"
ASSETS_DIR = REPO_ROOT / "assets"
POSTED_FILE = DATA_DIR / "posted.json"

LOG = logging.getLogger("affiliate")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str | None = None) -> None:
    """Configure root logging once, with a compact, greppable format."""
    lvl = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(lvl)
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                          datefmt="%Y-%m-%dT%H:%M:%S")
    )
    root.addHandler(handler)
    root.setLevel(lvl)


# ---------------------------------------------------------------------------
# .env loading (no external dependency)
# ---------------------------------------------------------------------------

def load_dotenv(path: Path | str | None = None) -> None:
    """Load KEY=VALUE pairs from a .env file without overwriting real env vars."""
    env_path = Path(path) if path else REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAPIDAPI_HOST = "axesso-axesso-amazon-data-service-v1.p.rapidapi.com"
AMAZON_BASE = "https://www.amazon.com"
CF_FLUX_MODEL = "@cf/black-forest-labs/flux-1-schnell"


@dataclass
class Config:
    """Runtime configuration read exclusively from the environment."""

    rapidapi_key: str = ""
    manylinks_api_key: str = ""
    cf_api_token: str = ""
    cf_account_id: str = ""
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_vision_model: str = "gpt-4o-mini"

    affiliate_tag: str = "bedroomassoci-20"
    niche: str = "bedroom & sleep products"
    target: str = "tiktok"
    posts_per_run: int = 1

    rapidapi_host: str = RAPIDAPI_HOST
    amazon_base: str = AMAZON_BASE

    # Budgets / guardrails
    max_search_calls: int = 10
    max_image_generations: int = 8
    max_api_keywords: int = 6
    min_score: float = 6.0
    min_rating: float = 4.2
    min_reviews: int = 500
    min_price: float = 10.0
    max_price: float = 80.0
    repeat_window_days: int = 90
    max_manylink_blocks: int = 30

    def missing_secrets(self) -> list[str]:
        required = {
            "RAPIDAPI_KEY": self.rapidapi_key,
            "MANYLINKS_API_KEY": self.manylinks_api_key,
            "CF_API_TOKEN": self.cf_api_token,
            "CF_ACCOUNT_ID": self.cf_account_id,
            "LLM_API_KEY": self.llm_api_key,
        }
        return [name for name, value in required.items() if not value]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    """Return the env var unless it is unset or blank, else ``default``."""
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def load_config() -> Config:
    return Config(
        rapidapi_key=os.environ.get("RAPIDAPI_KEY", "").strip(),
        manylinks_api_key=os.environ.get("MANYLINKS_API_KEY", "").strip(),
        cf_api_token=os.environ.get("CF_API_TOKEN", "").strip(),
        cf_account_id=os.environ.get("CF_ACCOUNT_ID", "").strip(),
        llm_api_key=os.environ.get("LLM_API_KEY", "").strip(),
        llm_base_url=_env_str("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        llm_model=_env_str("LLM_MODEL", "gpt-4o-mini"),
        llm_vision_model=_env_str("LLM_VISION_MODEL",
                                  _env_str("LLM_MODEL", "gpt-4o-mini")),
        affiliate_tag=_env_str("AFFILIATE_TAG", "bedroomassoci-20"),
        niche=_env_str("NICHE", "bedroom & sleep products"),
        target=_env_str("TARGET", "tiktok"),
        posts_per_run=max(1, _env_int("POSTS_PER_RUN", 1)),
        rapidapi_host=_env_str("RAPIDAPI_HOST", RAPIDAPI_HOST),
        amazon_base=_env_str("AMAZON_BASE", AMAZON_BASE).rstrip("/"),
    )


# ---------------------------------------------------------------------------
# HTTP with retry / backoff
# ---------------------------------------------------------------------------

class HttpError(RuntimeError):
    """Raised when an HTTP call ultimately fails."""

    def __init__(self, status: int, message: str, body: str = ""):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.body = body


def _retry_after_seconds(resp: requests.Response) -> float | None:
    value = resp.headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json_body: Any = None,
    data: Any = None,
    timeout: float = 60.0,
    retries: int = 4,
    backoff: float = 1.5,
    retry_statuses: Iterable[int] = (429, 500, 502, 503, 504),
) -> requests.Response:
    """Perform an HTTP request, retrying transient failures with backoff.

    Honors ``Retry-After`` on 429 responses. Raises :class:`HttpError` when all
    attempts are exhausted.
    """
    retry_statuses = tuple(retry_statuses)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
                data=data,
                timeout=timeout,
            )
        except requests.RequestException as exc:  # network level failure
            last_error = exc
            if attempt >= retries:
                break
            sleep_for = min(30.0, backoff ** (attempt + 1)) + random.uniform(0, 0.5)
            LOG.warning("request error (%s %s): %s - retrying in %.1fs",
                        method, _safe_url(url), exc, sleep_for)
            time.sleep(sleep_for)
            continue

        if resp.status_code in retry_statuses and attempt < retries:
            delay = _retry_after_seconds(resp)
            if delay is None:
                delay = min(30.0, backoff ** (attempt + 1)) + random.uniform(0, 0.5)
            LOG.warning("%s %s -> %s, retrying in %.1fs",
                        method, _safe_url(url), resp.status_code, delay)
            time.sleep(delay)
            continue

        if resp.status_code >= 400:
            body = resp.text[:500]
            raise HttpError(resp.status_code, resp.reason or "error", body)
        return resp

    raise HttpError(0, f"request failed: {last_error}")


def _safe_url(url: str) -> str:
    """Strip query strings so keys never leak into logs."""
    return url.split("?", 1)[0]


# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible chat completions)
# ---------------------------------------------------------------------------

class LLMError(RuntimeError):
    pass


class LLM:
    """Minimal OpenAI-compatible chat client used for reasoning and vision."""

    def __init__(self, cfg: Config):
        self.base_url = cfg.llm_base_url
        self.api_key = cfg.llm_api_key
        self.model = cfg.llm_model
        self.vision_model = cfg.llm_vision_model

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = request(
            "POST",
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json_body=payload,
            timeout=120.0,
            retries=2,
        )
        return resp.json()

    def chat(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.7,
        images: list[str] | None = None,
    ) -> str:
        """Return the assistant message content as text."""
        if not self.enabled:
            raise LLMError("LLM_API_KEY is not configured")

        content: Any = user
        if images:
            content = [{"type": "text", "text": user}]
            for img in images:
                content.append({"type": "image_url", "image_url": {"url": img}})

        payload: dict[str, Any] = {
            "model": model or self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = self._post(payload)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected LLM response shape: {exc}") from exc

    def chat_json(self, system: str, user: str, **kwargs: Any) -> Any:
        """Call :meth:`chat` in JSON mode and parse the result."""
        text = self.chat(system, user, json_mode=True, **kwargs)
        return parse_json_loose(text)

    def review_image(self, image_data_url: str, prompt: str) -> str:
        """Ask the vision model to critique a single frame."""
        return self.chat(
            "You are a meticulous TikTok creative reviewer. Reply with strict JSON.",
            prompt,
            model=self.vision_model,
            json_mode=True,
            images=[image_data_url],
        )


def parse_json_loose(text: str) -> Any:
    """Parse JSON that may be wrapped in prose or markdown fences."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```" in text:
        for chunk in text.split("```"):
            candidate = chunk.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") or candidate.startswith("["):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError("could not parse JSON from model output")


# ---------------------------------------------------------------------------
# JSON file helpers
# ---------------------------------------------------------------------------

def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOG.warning("could not read %s: %s", path, exc)
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# GitHub Actions step summary
# ---------------------------------------------------------------------------

def append_summary(markdown: str) -> None:
    """Append a block to $GITHUB_STEP_SUMMARY when running in Actions."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        LOG.info("step summary (no GITHUB_STEP_SUMMARY set):\n%s", markdown)
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(markdown.rstrip() + "\n")
    except OSError as exc:  # pragma: no cover - best effort only
        LOG.warning("could not write step summary: %s", exc)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def slugify(text: str, max_len: int = 60) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in " -_" else " " for ch in text)
    cleaned = "-".join(cleaned.split())
    return cleaned[:max_len].strip("-") or "product"


def shorten(text: str, max_len: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


@dataclass
class Guard:
    """Simple call-budget counter used to respect API quotas."""

    name: str
    limit: int
    used: int = 0
    events: list[str] = field(default_factory=list)

    def spend(self, n: int = 1) -> bool:
        if self.used + n > self.limit:
            LOG.warning("budget exhausted for %s (%d/%d)", self.name, self.used, self.limit)
            return False
        self.used += n
        return True

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)
