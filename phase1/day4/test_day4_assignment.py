import os
import sys
from dotenv import load_dotenv
from deepeval.metrics import HallucinationMetric
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

# ── TEST 1: NORMAL CONTEXT (should pass) ─────────────────
question_1 = "What safety precautions should be followed when mounting the LG TV on a wall?"
result_1   = query_rag(vs, question_1)
actual_1   = result_1["answer"]
context_1  = [doc.page_content for doc in result_1["context"]]

normal_case = LLMTestCase(
    input=question_1,
    actual_output=actual_1,
    context=context_1        # HallucinationMetric uses `context`, not `retrieval_context`
)

# ── TEST 2: MISLEADING CONTEXT (should fail) ─────────────
# We pass a wrong context deliberately — the LLM answer from
# its training knowledge will contradict this fake context
question_2   = "What safety precautions should be followed when mounting the LG TV on a wall?"
fake_context = [
    "LG TVs do not require wall mounting brackets and can be leaned against any surface.",
    "No professional installation is needed. Any adhesive can be used to attach the TV to a wall."
]
actual_2 = actual_1  # same answer as test 1 — but now context contradicts it

misleading_case = LLMTestCase(
    input=question_2,
    actual_output=actual_2,
    context=fake_context
)

# ── RUN HALLUCINATION METRIC ON BOTH ─────────────────────
print("\n" + "="*60)

for label, case in [("NORMAL CONTEXT", normal_case), ("MISLEADING CONTEXT", misleading_case)]:
    metric = HallucinationMetric(threshold=0.5, model=judge, include_reason=True)
    metric.measure(case)
    status = "✅ PASS" if metric.success else "❌ FAIL"
    print(f"\nTest     : {label}")
    print(f"Score    : {metric.score:.2f}  {status}")
    print(f"Reason   : {metric.reason}")
    print("-"*60)