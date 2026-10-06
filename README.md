# rag_evals

RAG pipeline over the LLM-evals course transcripts, with a DeepEval eval suite
(retriever, generator, end-to-end triad, safety, toxicity, leakage, scope, plus
operational latency/cost/reliability evals and a regression harness).

This copy swaps the upstream OpenAI-only stack for a configurable one:

| Piece | Upstream | Here |
|---|---|---|
| Answering LLM | `ChatOpenAI("gpt-4o-mini")` | any OpenAI-compatible endpoint — **Groq** by default, also xAI Grok / OpenAI |
| DeepEval judge | model strings (`"gpt-4o-mini"`) | a `DeepEvalBaseLLM` wrapper around the same endpoint |
| Embeddings | `OpenAIEmbeddings("text-embedding-3-large")` | **HuggingFace** — local sentence-transformers, or the HF Inference API |
| Cost pricing | hardcoded gpt-4o-mini rates | `PRICE_*` env vars |

All of it lives in [src/config.py](src/config.py); the rest of the project just
calls `get_llm()`, `get_embeddings()` and `get_judge()`.

## Setup

Everything — virtualenv, pip cache, and downloaded HuggingFace models — stays on
the D: drive; nothing is written to `C:\Users\...`.

```powershell
py -3.13 -m venv .venv                     # Python 3.13 (torch has no 3.14 wheels yet)
.\.venv\Scripts\Activate.ps1

$env:PIP_CACHE_DIR = "D:\ai\rag_eval\.pip-cache"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .

copy .env.example .env                     # then fill in GROQ_API_KEY
```

torch is installed from the CPU index on purpose — the default wheel bundles
CUDA and costs ~2.5GB. Want GPU embeddings instead? Reinstall torch from the
`cu121`/`cu124` index and set `EMBEDDING_DEVICE=cuda` in `.env`.

`HF_HOME=./.cache/huggingface` in `.env` keeps the embedding + reranker model
downloads inside the project folder rather than the default cache on C:.

Minimum you must edit in `.env`:

```ini
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...              # https://console.groq.com/keys
CHAT_MODEL=openai/gpt-oss-120b
JUDGE_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_MODE=local              # "api" to call HF instead of running locally
```

Check the wiring before anything else — this prints the resolved config, makes
one chat call, and embeds one string:

```bash
python -m src.config              # resolved config + a live chat and embed call
python -m src.config --list-models   # what your key can actually call
```

## Using a different provider

* **xAI Grok** — `LLM_PROVIDER=xai`, `XAI_API_KEY=...`, `CHAT_MODEL=grok-4-fast`
* **OpenAI** — `LLM_PROVIDER=openai`, `OPENAI_API_KEY=...`, `CHAT_MODEL=gpt-4o-mini`
* **Anything else** (Ollama, OpenRouter, Together, vLLM) — keep any provider and
  set `LLM_BASE_URL` + `LLM_API_KEY` to override the endpoint.

## Embeddings

`EMBEDDING_MODE=local` downloads the model once (~130MB for `bge-small`) and
runs it on CPU — no key, no per-call cost. `EMBEDDING_MODE=api` calls the HF
Inference API instead and needs `HUGGINGFACEHUB_API_TOKEN`.

Chroma cannot mix vector dimensions in one collection, so **if you change
`EMBEDDING_MODEL`, also change `CHROMA_DIR`** (or delete the old directory).
The store is rebuilt automatically on the next run.

## Running

```bash
python -m src.retriever            # build/query the vector store
python -m src.rag_pipeline         # retrieve -> rerank -> generate, one query
streamlit run src/app.py           # chat UI (run from the project root)

python -m evals.eval_retriever     # single evals
python -m evals.eval_rag_pipeline
python -m evals.eval_safety
python -m evals.run_suite          # the whole suite + regression verdict
python -m evals.eval_ops           # latency / cost / reliability
```

DeepEval may still ask for its own key on first run (`deepeval login`) for the
Confident AI dashboard — that is optional, local runs work without it.

## Running on a free tier (read this before the first eval)

Groq's on-demand tier allows **8,000 tokens per minute**, and one retrieval
judgement sends ~2.5k tokens of context. DeepEval's defaults fight that hard, so
three things are pre-configured here:

| Problem | Symptom | Setting |
|---|---|---|
| 20 parallel judge calls | `429 rate_limit_exceeded` | `EVAL_MAX_CONCURRENT=1`, `EVAL_THROTTLE=1` |
| 180s budget per test case | bare `TimeoutError` mid-run | `DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=900` |
| No backoff on 429 | run dies on the first limit | `LLM_MAX_RETRIES=8` |

The cost is wall-clock: a 15-golden retriever eval takes **20–30 minutes**
serialised, most of it waiting out the TPM window. On a paid tier, raise
`EVAL_MAX_CONCURRENT` to 5–10 and drop `EVAL_THROTTLE` to 0.

Two more Windows-specific fixes are baked in: DeepEval's on-disk test-case cache
is disabled (`CacheConfig(write_cache=False)` — its file locks collide with its
own async execution and crash the run), and stdout is forced to UTF-8 in
`src/config.py` (the rich reports contain emoji that cp1252 cannot encode).


## Notes on the judge

DeepEval builds its own client from a model *string*, which only works for the
providers it ships. `get_judge()` returns a `DeepEvalBaseLLM` instead, so the
metrics judge through your configured endpoint. It asks for native structured
output first and falls back to "reply with JSON, then parse it" for open models
whose tool-calling is unreliable — a bigger judge (`gpt-oss-120b`, not a 20B)
gives noticeably steadier scores.

Hosted catalogues move: Groq retired `llama-3.3-70b-versatile`, and a retired
model shows up as `404 model_not_found` from the first judge call. To see what
your key can actually call:

```powershell
.\.venv\Scripts\python.exe -m src.config --list-models
```

Judge scores are not comparable across judge models: if you switch
`JUDGE_MODEL`, regenerate the regression baseline rather than diffing against
one produced by a different judge.
