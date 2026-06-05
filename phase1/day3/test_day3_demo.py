import os
import sys
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

judge = ClaudeJudge()

# ── LOAD PIPELINE ────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from phase1.day1.rag_pipeline import load_documents, chunk_documents, build_vectorstore, query_rag

doc_paths = [
    "phase1/day1/docs/doc1.pdf",
    "phase1/day1/docs/doc2.pdf",
    "phase1/day1/docs/doc3.pdf"
]
docs   = load_documents(doc_paths)
chunks = chunk_documents(docs)
vs     = build_vectorstore(chunks)

# ── Q1 TEST CASE ─────────────────────────────────────────
# question = "What safety precautions should be followed when mounting the LG TV on a wall?"
# expected = "Use a solid wall, hire a professional installer, connect cables before mounting, remove the stand first."

question = "What should I do if the LG TV turns off suddenly and what are the possible causes?"
expected = "Check power control settings. The auto-off function turns off the TV after 15 minutes without input signal. Check if auto-off is activated in time settings."

result  = query_rag(vs, question)
actual  = result["answer"]
context = [doc.page_content for doc in result["context"]]

test_case = LLMTestCase(
    input=question,
    actual_output=actual,
    expected_output=expected,
    retrieval_context=context
)

# ── RUN ALL 4 METRICS ────────────────────────────────────
metrics = [
    AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=True),
    FaithfulnessMetric(threshold=0.7, model=judge, include_reason=True),
    ContextualPrecisionMetric(threshold=0.7, model=judge, include_reason=True),
    ContextualRecallMetric(threshold=0.7, model=judge, include_reason=True),
]

print("\n" + "="*60)
print(f"Question: {question}")
print("="*60)

for metric in metrics:
    metric.measure(test_case)
    status = "✅ PASS" if metric.success else "❌ FAIL"
    print(f"\n{metric.__class__.__name__}")
    print(f"Score  : {metric.score:.2f}  {status}")
    print(f"Reason : {metric.reason}")

print("\n" + "="*60)