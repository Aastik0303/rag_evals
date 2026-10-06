# eval_rag_pipeline.py
import os

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from src.rag_pipeline import RagPipeline
from evals.harness import load_goldens, summarize_by_metric, print_summary

from src.config import get_judge

load_dotenv()

# DeepEval's on-disk test-case cache uses file locks that collide with its own
# parallel async execution on Windows -- the lock fails, the cache object comes
# back None and the run dies mid-suite. We don't reuse cached judgements here
# anyway, so turn the cache off.
NO_CACHE = CacheConfig(write_cache=False)

# DeepEval fires 20 judge calls at once by default, which blows straight through
# a free-tier token-per-minute budget (Groq allows 8k TPM, and one retrieval
# judgement is ~3k tokens). Concurrency and the pause between tasks are env
# knobs -- raise EVAL_MAX_CONCURRENT once you are on a paid tier.
EVAL_ASYNC = AsyncConfig(
    max_concurrent=int(os.getenv("EVAL_MAX_CONCURRENT", "1")),
    throttle_value=float(os.getenv("EVAL_THROTTLE", "1")),
)

GOLDEN_PATH = "goldens/faithfulness_dataset.json"   # reuse the queries
# judge model name comes from JUDGE_MODEL in .env (was 'gpt-4o-mini' upstream)
JUDGE_MODEL = get_judge()
THRESHOLD = 0.7


def run(rag):
    # 1. LOAD queries (we only need the queries --- context comes from the pipeline now)
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED PIPELINE per query, build a test case from LIVE output
    test_cases = []
    for g in goldens:
        result = rag.invoke(g["query"])          # retrieve -> rerank -> generate

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=result["answer"],       # what the generator produced
                retrieval_context=result["context"],  # what the RETRIEVER returned
            )
        )

    # 3. THE THREE TRIAD METRICS
    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    # 4. EVALUATE
    result = evaluate(cache_config=NO_CACHE,
        async_config=EVAL_ASYNC, test_cases=test_cases, metrics=metrics)
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("rag_pipeline", run_local())