import os

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

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

# --- Test case 1: a good answer (should PASS) ---
case_1 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
)

# --- Test case 2: an off-topic answer (should FAIL) ---
case_2 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="France is a beautiful country famous for its food and wine.",
)

# --- One metric, judged by an LLM (pinned for reproducibility) ---
metric = AnswerRelevancyMetric(threshold=0.7, model=get_judge(), include_reason=True)

# --- Run BOTH cases through the metric, with a printed report ---
evaluate(cache_config=NO_CACHE,
        async_config=EVAL_ASYNC, test_cases=[case_1, case_2], metrics=[metric])