import os
import sys
import pytest
from dotenv import load_dotenv
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_anthropic import ChatAnthropic

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from phase1.day1.rag_pipeline import load_documents, chunk_documents, build_vectorstore, query_rag

# ── CLAUDE AS JUDGE ──────────────────────────────────────
class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=0,
            max_tokens=1024
        )
    def load_model(self): return self.model
    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content
    async def a_generate(self, prompt: str) -> str:
        result = await self.model.ainvoke(prompt)
        return result.content
    def get_model_name(self): return "claude-haiku-4-5-20251001;"

judge = ClaudeJudge()

# ── FIXTURES ─────────────────────────────────────────────
@pytest.fixture(scope="session")
def vectorstore():
    doc_paths = [
        "phase1/day1/docs/doc1.pdf",
        "phase1/day1/docs/doc2.pdf",
        "phase1/day1/docs/doc3.pdf"
    ]
    docs   = load_documents(doc_paths)
    chunks = chunk_documents(docs)
    return build_vectorstore(chunks)

# ── TEST DATA ─────────────────────────────────────────────
test_data = [
    {
        "id": "Q1",
        "question": "What safety precautions should be followed when mounting the LG TV on a wall?",
        "expected": "Use a solid wall, hire a professional installer, connect cables before mounting, remove the stand first."
    },
    {
        "id": "Q3",
        "question": "How should I clean the Sony headphones and what materials should I avoid?",
        "expected": "Wipe with a soft dry cloth. For heavy dirt use diluted neutral detergent. Avoid thinner, benzene, and alcohol."
    },
    {
        "id": "Q4",
        "question": "What should I do if the LG TV turns off suddenly and what are the possible causes?",
        "expected": "Check power control settings. The auto-off function turns off the TV after 15 minutes without input signal. Check if auto-off is activated in time settings."
    },
    {
        "id": "Q5",
        "question": "What are the battery safety warnings mentioned across any of the three device manuals?",
        "expected": "Use only approved charger and cable. Do not charge with wet jack. Operating temperature 0-35C. Sony headset flashes red when battery is low."
    }
]

THRESHOLDS = [0.5, 0.7, 0.9]

METRICS = [
    ("AnswerRelevancy",    AnswerRelevancyMetric),
    ("Faithfulness",       FaithfulnessMetric),
    ("ContextualPrecision",ContextualPrecisionMetric),
    ("ContextualRecall",   ContextualRecallMetric),
]

# ── HELPER ───────────────────────────────────────────────
def build_test_case(vs, item):
    result  = query_rag(vs, item["question"])
    actual  = result["answer"]
    context = [doc.page_content for doc in result["context"]]
    return LLMTestCase(
        input=item["question"],
        actual_output=actual,
        expected_output=item["expected"],
        retrieval_context=context,
        context=context
    )

# ── SCORE COLLECTION ─────────────────────────────────────
results = {}

def get_scores(vs, item):
    key = item["id"]
    if key not in results:
        tc = build_test_case(vs, item)
        scores = {}
        for name, MetricClass in METRICS:
            m = MetricClass(threshold=0.7, model=judge, include_reason=False)
            m.measure(tc)
            scores[name] = round(m.score, 2)
        results[key] = scores
    return results[key]

# ── PARAMETRIZED TESTS ───────────────────────────────────
@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("item", test_data, ids=[d["id"] for d in test_data])
@pytest.mark.parametrize("metric_name,MetricClass", METRICS, ids=[m[0] for m in METRICS])
def test_threshold(vectorstore, metric_name, MetricClass, item, threshold):
    scores = get_scores(vectorstore, item)
    score  = scores[metric_name]
    passed = score >= threshold
    print(f"\n[{item['id']}][{metric_name}][t={threshold}] score={score} {'✅' if passed else '❌'}")
    assert passed, f"[{item['id']}] {metric_name} score {score} below threshold {threshold}"

# ── SUMMARY TABLE (printed after all tests) ──────────────
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not results:
        return
    print("\n\n" + "="*70)
    print("THRESHOLD COMPARISON TABLE")
    print("="*70)
    header = f"{'Question':<6} {'Metric':<22} {'Score':<8} {'t=0.5':<8} {'t=0.7':<8} {'t=0.9':<8}"
    print(header)
    print("-"*70)
    for qid, scores in sorted(results.items()):
        for metric, score in scores.items():
            p05 = "✅" if score >= 0.5 else "❌"
            p07 = "✅" if score >= 0.7 else "❌"
            p09 = "✅" if score >= 0.9 else "❌"
            print(f"{qid:<6} {metric:<22} {score:<8} {p05:<8} {p07:<8} {p09:<8}")
        print("-"*70)