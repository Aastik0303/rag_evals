"""
src/config.py — one place where the project's models come from.

The upstream course repo was hardwired to OpenAI (OpenAIEmbeddings +
ChatOpenAI("gpt-4o-mini") + string judge names like "gpt-4o-mini"). This module
replaces that with a configurable stack driven entirely by .env:

  * CHAT / JUDGE  -> any OpenAI-compatible endpoint. Defaults to Groq
                     (https://api.groq.com/openai/v1), and also supports xAI
                     Grok and plain OpenAI by switching LLM_PROVIDER.
  * EMBEDDINGS    -> HuggingFace. Local sentence-transformers by default (free,
                     no key), or the HF Inference API if you'd rather not run
                     the model on your machine.

Everything else in the project imports from here:

    from src.config import get_llm, get_embeddings, get_judge, CHROMA_DIR
"""

import os
import json
import re
import sys

from dotenv import load_dotenv

load_dotenv()

# DeepEval prints its reports through rich, emoji included. On Windows the
# console defaults to cp1252, which cannot encode them -- the suite then dies
# with UnicodeEncodeError before a single metric is reported. Force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Keep HuggingFace model downloads inside the project instead of the default
# %USERPROFILE%\.cache on C:. huggingface_hub reads these at import time, so we
# set them here -- config is imported before anything that pulls a model.
_HF_HOME = os.path.abspath(os.getenv("HF_HOME") or ".cache/huggingface")
os.environ["HF_HOME"] = _HF_HOME
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_HF_HOME, "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _HF_HOME)


# ============================================================
# 1. PROVIDER TABLE
# ============================================================
# Every provider here speaks the OpenAI wire format, so a single ChatOpenAI
# client covers all of them -- only base_url / api key / model names differ.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": ("GROQ_API_KEY", "LLM_API_KEY"),
        "default_model": "openai/gpt-oss-120b",
    },
    "xai": {                       # xAI's Grok
        "base_url": "https://api.x.ai/v1",
        "key_env": ("XAI_API_KEY", "GROK_API_KEY", "LLM_API_KEY"),
        "default_model": "grok-4-fast",
    },
    "openai": {
        "base_url": None,          # the SDK default
        "key_env": ("OPENAI_API_KEY", "LLM_API_KEY"),
        "default_model": "gpt-4o-mini",
    },
}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
if LLM_PROVIDER not in PROVIDERS:
    raise ValueError(
        f"LLM_PROVIDER={LLM_PROVIDER!r} is not supported. "
        f"Choose one of: {', '.join(PROVIDERS)}"
    )

_P = PROVIDERS[LLM_PROVIDER]

# The answering model (src/generator.py) and the judge model (evals/) are kept
# separate on purpose: judging with the same model that answered biases the
# score, and you often want a cheaper/faster model on one side than the other.
CHAT_MODEL = os.getenv("CHAT_MODEL", _P["default_model"])
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL", CHAT_MODEL)

LLM_BASE_URL = os.getenv("LLM_BASE_URL") or _P["base_url"]
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "8"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))

# DeepEval gives each test case a 180s budget by default. On a rate-limited free
# tier a single judgement can spend minutes just waiting out 429 backoff, so the
# budget has to be much larger or the whole run dies with a bare TimeoutError.
os.environ.setdefault(
    "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE",
    os.getenv("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "900"),
)

# Vector store location. Different embedding models produce different vector
# dimensions, so if you switch EMBEDDING_MODEL you must point this somewhere new
# (or delete the old directory) -- Chroma cannot mix dimensions in one collection.
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_store")

# HuggingFace embeddings. "local" runs the model on your machine through
# sentence-transformers (free, offline after the first download); "api" calls
# the HF Inference API and needs HUGGINGFACEHUB_API_TOKEN.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "local").strip().lower()
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")


def _api_key() -> str:
    for name in _P["key_env"]:
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError(
        f"No API key found for LLM_PROVIDER={LLM_PROVIDER}. "
        f"Set one of {', '.join(_P['key_env'])} in your .env "
        f"(copy .env.example to .env to get started)."
    )


# ============================================================
# 2. CHAT MODEL
# ============================================================
def get_llm(model: str | None = None, temperature: float | None = None):
    """A LangChain chat model pointed at the configured provider."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or CHAT_MODEL,
        temperature=CHAT_TEMPERATURE if temperature is None else temperature,
        api_key=_api_key(),
        base_url=LLM_BASE_URL,
        # free tiers are tight on tokens-per-minute; the OpenAI client backs off
        # and honours the provider's Retry-After header on 429s
        max_retries=LLM_MAX_RETRIES,
        timeout=LLM_TIMEOUT,
    )


# ============================================================
# 3. EMBEDDINGS (HuggingFace)
# ============================================================
_embeddings = None


def get_embeddings():
    """
    HuggingFace embeddings, built once and reused.

    Local models are loaded into memory, so caching matters: the retriever, the
    export script and the evals all end up calling this.
    """
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    if EMBEDDING_MODE == "api":
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "EMBEDDING_MODE=api needs HUGGINGFACEHUB_API_TOKEN in your .env."
            )
        _embeddings = HuggingFaceEndpointEmbeddings(
            model=EMBEDDING_MODEL,
            huggingfacehub_api_token=token,
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            # Chroma scores with cosine distance -- normalising makes the dot
            # product the cosine, which is what these models are trained for.
            encode_kwargs={"normalize_embeddings": True},
        )

    return _embeddings


# ============================================================
# 4. JUDGE (DeepEval custom model)
# ============================================================
# DeepEval only knows how to build its own OpenAI/Anthropic/... clients from a
# model *string*. Any other provider has to arrive as a DeepEvalBaseLLM object,
# which is what this wrapper is: it hands DeepEval the same OpenAI-compatible
# client the rest of the project uses.
#
# The metrics ask for structured output (a pydantic schema). We try native
# structured output first and fall back to "reply with JSON, then parse it",
# because not every open model on Groq supports tool-calling reliably.
def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return text.strip()
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return text[start:end + 1].strip() if end > start else text.strip()


def _as_json_prompt(prompt: str, schema) -> str:
    """Ask for raw JSON matching the schema, for models without tool-calling."""
    try:
        spec = json.dumps(schema.model_json_schema())
    except Exception:
        spec = str(schema)
    return (
        f"{prompt}\n\n"
        "Respond with ONLY a single JSON object that validates against this "
        f"JSON schema. No prose, no markdown fences.\n{spec}"
    )


def _build_judge_class():
    from deepeval.models import DeepEvalBaseLLM

    class CustomJudge(DeepEvalBaseLLM):
        """DeepEval judge backed by any OpenAI-compatible endpoint."""

        def __init__(self, model_name: str | None = None):
            self.model_name = model_name or JUDGE_MODEL_NAME
            self._client = None
            super().__init__(self.model_name)

        def load_model(self):
            if self._client is None:
                # judges must be deterministic -- temperature pinned to 0
                self._client = get_llm(model=self.model_name, temperature=0)
            return self._client

        def get_model_name(self) -> str:
            return f"{LLM_PROVIDER}:{self.model_name}"

        # --- sync -------------------------------------------------------
        def generate(self, prompt: str, schema=None, **kwargs):
            client = self.load_model()
            if schema is None:
                return client.invoke(prompt).content
            try:
                return client.with_structured_output(schema).invoke(prompt)
            except Exception:
                message = client.invoke(_as_json_prompt(prompt, schema))
                return self._json_fallback(message.content, schema)

        # --- async ------------------------------------------------------
        async def a_generate(self, prompt: str, schema=None, **kwargs):
            client = self.load_model()
            if schema is None:
                return (await client.ainvoke(prompt)).content
            try:
                return await client.with_structured_output(schema).ainvoke(prompt)
            except Exception:
                message = await client.ainvoke(_as_json_prompt(prompt, schema))
                return self._json_fallback(message.content, schema)

        @staticmethod
        def _json_fallback(text: str, schema):
            return schema.model_validate(json.loads(_extract_json(text)))

    return CustomJudge


_judge_cache: dict = {}


def get_judge(model: str | None = None):
    """
    The judge object to pass to DeepEval metrics:

        AnswerRelevancyMetric(threshold=0.7, model=get_judge())

    Cached per model name so a suite running many metrics reuses one client.
    """
    name = model or JUDGE_MODEL_NAME
    if name not in _judge_cache:
        _judge_cache[name] = _build_judge_class()(name)
    return _judge_cache[name]


def list_models() -> list[str]:
    """
    Model ids your key can actually call, straight from the provider.

    Worth running whenever you hit a 404 `model_not_found`: hosted catalogues
    are not stable -- Groq in particular retires models with little notice, so a
    name that worked last month may simply be gone.
    """
    from openai import OpenAI

    client = OpenAI(api_key=_api_key(), base_url=LLM_BASE_URL)
    return sorted(m.id for m in client.models.list().data)


# quick sanity check:  python -m src.config
# available models:    python -m src.config --list-models
if __name__ == "__main__":
    import sys

    if "--list-models" in sys.argv:
        print(f"models available to your {LLM_PROVIDER} key:")
        for model_id in list_models():
            print(f"  {model_id}")
        sys.exit(0)

    print(f"provider    : {LLM_PROVIDER}  ({LLM_BASE_URL or 'default base_url'})")
    print(f"chat model  : {CHAT_MODEL}")
    print(f"judge model : {JUDGE_MODEL_NAME}")
    print(f"embeddings  : {EMBEDDING_MODEL} ({EMBEDDING_MODE})")
    print(f"chroma dir  : {CHROMA_DIR}")

    print("\n-- chat --")
    print(get_llm().invoke("Reply with exactly: ok").content)

    print("\n-- embeddings --")
    print(f"dimension: {len(get_embeddings().embed_query('hello'))}")
