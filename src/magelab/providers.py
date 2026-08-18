"""
Non-Anthropic models, reached through an Anthropic-shaped gateway.

The Claude Agent SDK speaks the Anthropic Messages API and nothing else. To run
an agent on a non-Anthropic model we do not replace the runner — we point that
agent's Claude Code subprocess at a local proxy that accepts Anthropic-shaped
requests and forwards them to the other provider. Because
`ClaudeAgentOptions.env` is per-agent, this is decided per agent: a single org
can hold Claude agents talking to api.anthropic.com and gateway agents talking
to the proxy, with the same tools, the same prompts and the same wire.

What this buys, versus a second AgentRunner implementation: the agent keeps
Claude Code's native tools (Read/Grep/Glob/Write/Edit), its session and resume
handling, and its transcript stream. The only variable across the assembly is
the model — which is what a cross-model comparison wants.

What it costs: parameters with no Anthropic equivalent (an OpenAI reasoning
model's `reasoning_effort`, say) cannot be reached through the Messages API
shape, and `ResultMessage.total_cost_usd` is computed by Claude Code from its
own Anthropic price table, so it is meaningless for a gateway model. See
`cost_from_usage` for how that is handled.

Configuration (all optional, read from the environment):

    MAGELAB_GATEWAY_URL     base URL of the proxy   (default http://127.0.0.1:4000)
    MAGELAB_GATEWAY_TOKEN   token the proxy expects (default sk-magelab-local)
    MAGELAB_MODEL_PRICES    JSON, USD per million tokens, keyed by model id:
                            {"<model>": {"input": .., "output": .., "cached_input": ..}}
                            `cached_input` is optional and falls back to `input`.
"""

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_GATEWAY_URL = "http://127.0.0.1:4000"
DEFAULT_GATEWAY_TOKEN = "sk-magelab-local"

# Model ids that are not Anthropic's and therefore need the gateway. Kept as a
# pattern rather than a list so a new OpenAI model works without a code change;
# anything unrecognised is assumed to be Anthropic and left alone.
_GATEWAY_MODEL = re.compile(r"^(gpt-|chatgpt-|o[1-4](-|$)|sora-|davinci-|babbage-)", re.IGNORECASE)

_warned_missing_price: set[str] = set()


def is_gateway_model(model: Optional[str]) -> bool:
    """True if `model` must be reached through the gateway rather than Anthropic."""
    return bool(model) and bool(_GATEWAY_MODEL.match(model or ""))


def gateway_url() -> str:
    """Base URL of the Anthropic-shaped proxy."""
    return os.environ.get("MAGELAB_GATEWAY_URL", DEFAULT_GATEWAY_URL).rstrip("/")


def gateway_token() -> str:
    """Token the proxy expects. Never the upstream provider's key — the proxy holds that."""
    return os.environ.get("MAGELAB_GATEWAY_TOKEN", DEFAULT_GATEWAY_TOKEN)


def gateway_env(model: Optional[str]) -> dict[str, str]:
    """
    Environment for one agent's Claude Code subprocess, or {} for Anthropic models.

    Returned keys are merged over the caller's env, so the Anthropic key the
    runner would otherwise inject is deliberately overwritten here: the upstream
    provider's credential lives in the proxy's environment, and the real
    Anthropic key must not be sent to a third-party endpoint.
    """
    if not is_gateway_model(model):
        return {}
    token = gateway_token()
    return {
        "ANTHROPIC_BASE_URL": gateway_url(),
        # Claude Code sends whichever of these it finds; both are the proxy's
        # token so the real Anthropic key cannot leak to it.
        "ANTHROPIC_AUTH_TOKEN": token,
        "ANTHROPIC_API_KEY": token,
        # Claude Code reaches for a small model for incidental work (titles,
        # summaries). That id does not exist behind the gateway, so pin it to
        # the model we are actually running.
        "ANTHROPIC_SMALL_FAST_MODEL": model or "",
        # No telemetry or auto-update traffic to Anthropic for a run that is not
        # using Anthropic.
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1",
        "DISABLE_TELEMETRY": "1",
    }


def effective_prices(models: Optional[list[str]] = None) -> dict[str, dict[str, float]]:
    """The configured prices, optionally narrowed to `models`.

    Exposed so a caller can record what a run was actually costed at. A price
    that lives only in an operator's shell is not auditable six months later —
    stamping it beside the cost makes the number reproducible, and makes an
    later correction to the table visibly a different number rather than a
    silently incomparable one.
    """
    prices = _prices()
    if models is None:
        return prices
    return {m: prices[m] for m in dict.fromkeys(models) if m in prices}


def _prices() -> dict[str, dict[str, float]]:
    raw = os.environ.get("MAGELAB_MODEL_PRICES", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("MAGELAB_MODEL_PRICES is not valid JSON, ignoring: %s", e)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("MAGELAB_MODEL_PRICES must be an object keyed by model id, ignoring")
        return {}
    return parsed


def cost_from_usage(model: Optional[str], usage: Any) -> Optional[float]:
    """
    Cost in USD for a gateway model's turn, or None if it cannot be known.

    Claude Code prices every run against its own Anthropic table, so the
    `total_cost_usd` it reports for a gateway model is wrong rather than
    approximate. Rather than record a wrong number in a run's provenance, we
    recompute from token counts when a price is configured and return None when
    it is not — an absent cost is honest, a fabricated one is not.
    """
    if not is_gateway_model(model):
        return None
    price = _prices().get(model or "")
    if not price:
        if model and model not in _warned_missing_price:
            _warned_missing_price.add(model)
            logger.warning(
                "No price configured for gateway model '%s' — cost will be recorded as unknown. "
                "Set MAGELAB_MODEL_PRICES to record it, e.g. "
                '\'{"%s": {"input": 1.25, "output": 10.0}}\' (USD per million tokens).',
                model,
                model,
            )
        return None
    if not isinstance(usage, dict):
        return None

    def _tokens(*names: str) -> int:
        for n in names:
            v = usage.get(n)
            if isinstance(v, (int, float)):
                return int(v)
        return 0

    # Anthropic's convention, which is what the gateway's responses are shaped
    # to: input_tokens counts only what was NOT served from cache, and cache
    # reads are reported separately. They are billed far more cheaply — $0.02
    # against $0.20 per MTok on gpt-5.6-luna — so charging them as ordinary
    # input would overstate a long deliberation, where each turn re-sends the
    # whole accumulated thread.
    fresh_input = _tokens("input_tokens", "prompt_tokens")
    cached_input = _tokens("cache_read_input_tokens", "cached_tokens")
    cache_writes = _tokens("cache_creation_input_tokens")
    output_tokens = _tokens("output_tokens", "completion_tokens")
    try:
        rate_in = float(price["input"])
        rate_out = float(price["output"])
        # No separate cached rate configured: price cache reads as input, which
        # over-counts rather than under-counts.
        rate_cached = float(price.get("cached_input", rate_in))
    except (KeyError, TypeError, ValueError):
        logger.warning("Price entry for '%s' needs numeric 'input' and 'output' keys — ignoring", model)
        return None
    return (
        ((fresh_input + cache_writes) / 1_000_000) * rate_in
        + (cached_input / 1_000_000) * rate_cached
        + (output_tokens / 1_000_000) * rate_out
    )


def check_gateway_reachable(timeout: float = 5.0) -> Optional[str]:
    """
    None if the gateway answers, else a human-readable reason it did not.

    Called before a run so that a proxy that is not running fails in the first
    second rather than at the first agent dispatch, ten minutes in.
    """
    import urllib.error
    import urllib.request

    url = f"{gateway_url()}/health/liveliness"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {gateway_token()}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status < 400:
                return None
            return f"{url} returned HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        # Any HTTP answer means something is listening and speaking HTTP, which
        # is all this check is for; auth is exercised by the run itself.
        if e.code in (401, 403, 404):
            return None
        return f"{url} returned HTTP {e.code}"
    except Exception as e:
        return f"could not reach {url} ({e.__class__.__name__}: {e})"
