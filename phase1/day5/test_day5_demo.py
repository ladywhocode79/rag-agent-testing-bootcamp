import os
import sys
import pytest
from dotenv import load_dotenv
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric
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
    def get_model_name(self): return "claude-haiku-4-5-20251001"

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

@pytest.fixture(scope="session")
def judge():
    return ClaudeJudge()

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

# ── HELPER ───────────────────────────────────────────────
def build_test_case(vectorstore, item):
    result  = query_rag(vectorstore, item["question"])
    actual  = result["answer"]
    context = [doc.page_content for doc in result["context"]]
    return LLMTestCase(
        input=item["question"],
        actual_output=actual,
        expected_output=item["expected"],
        retrieval_context=context,
        context=context
    )

# ── TESTS ─────────────────────────────────────────────────
@pytest.mark.parametrize("item", test_data, ids=[d["id"] for d in test_data])
def test_answer_relevancy(vectorstore, judge, item, capsys):
    tc     = build_test_case(vectorstore, item)
    metric = AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=True)
    metric.measure(tc)
    output = f"[{item['id']}] AnswerRelevancy: {metric.score:.2f} — {metric.reason}"
    print(output, flush=True)
    assert metric.success, output

@pytest.mark.parametrize("item", test_data, ids=[d["id"] for d in test_data])
def test_faithfulness(vectorstore, judge, item, capsys):
    tc     = build_test_case(vectorstore, item)
    metric = FaithfulnessMetric(threshold=0.7, model=judge, include_reason=True)
    metric.measure(tc)
    output = f"[{item['id']}] Faithfulness: {metric.score:.2f} — {metric.reason}"
    print(output, flush=True)
    assert metric.success, output

@pytest.mark.parametrize("item", test_data, ids=[d["id"] for d in test_data])
def test_hallucination(vectorstore, judge, item, capsys):
    tc = build_test_case(vectorstore, item)
    metric = HallucinationMetric(threshold=0.5, model=judge, include_reason=True)
    metric.measure(tc)
    item_id = item["id"]
    output = f"[{item_id}] Hallucination: {metric.score:.2f} — {metric.reason}"
    print(output, flush=True)
    assert metric.score < metric.threshold, output

@pytest.mark.parametrize("item", test_data, ids=[d["id"] for d in test_data])
def test_contextual_recall(vectorstore, judge, item, capsys):
    tc     = build_test_case(vectorstore, item)
    metric = ContextualRecallMetric(threshold=0.7, model=judge, include_reason=True)
    metric.measure(tc)
    output = f"[{item['id']}] ContextualRecall: {metric.score:.2f} — {metric.reason}"
    print(output, flush=True)
    assert metric.success, output

def test_summary_report(vectorstore, judge):
    """Generate comprehensive HTML report with all metrics"""
    print("\n" + "="*80)
    print("RAG PIPELINE EVALUATION SUMMARY")
    print("="*80)

    for item in test_data:
        tc = build_test_case(vectorstore, item)
        print(f"\n[{item['id']}] {item['question']}")
        print("-" * 80)

        metrics = [
            ("AnswerRelevancy", AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=True)),
            ("Faithfulness", FaithfulnessMetric(threshold=0.7, model=judge, include_reason=True)),
            ("Hallucination", HallucinationMetric(threshold=0.5, model=judge, include_reason=True)),
            ("ContextualRecall", ContextualRecallMetric(threshold=0.7, model=judge, include_reason=True)),
        ]

        for metric_name, metric in metrics:
            metric.measure(tc)
            status = "✓ PASS" if metric.success else "✗ FAIL"
            print(f"{metric_name:20s}: {metric.score:.2f} — {status}")
            print(f"  Reason: {metric.reason}\n")

    print("="*80)

