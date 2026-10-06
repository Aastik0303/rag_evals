import os
import json

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from src.reranker import RerankingRetriever

from src.config import get_judge, EMBEDDING_MODEL

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

GOLDEN_PATH = "goldens/retriever_goldens.json"
# judge model name comes from JUDGE_MODEL in .env (was 'gpt-4.1-mini' upstream)
JUDGE_MODEL = get_judge()
THRESHOLD = 0.7


# 1. LOAD the golden set --- the fixed, human-authored truth
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


# 2. RUN THE RETRIEVER on each question to fill retrieval_context,
#    then build one test case per golden.
retriever = RerankingRetriever()

test_cases = []

for g in goldens:
    retrieved = retriever.invoke(g["query"])
    retrieval_context = [doc.page_content for doc in retrieved]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="(generator not evaluated in this run)",
        )
    )


# 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
metrics = [
    ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]


# 4. EVALUATE --- every metric on every case, batched + parallel, with a printed report
evaluate(
    cache_config=NO_CACHE,
        async_config=EVAL_ASYNC,
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
        "retriever": "base_k5",          # vs "reranked" when you swap it in
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "judge_model": JUDGE_MODEL.get_model_name(),
        "golden_set": GOLDEN_PATH,
    },
)