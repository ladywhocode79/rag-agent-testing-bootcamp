import os
import sys
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_anthropic import ChatAnthropic

load_dotenv()

# ── 1. CLAUDE AS JUDGE ───────────────────────────────────
class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=0,
            max_tokens=1024
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        result = await self.model.ainvoke(prompt)
        return result.content

    def get_model_name(self) -> str:
        return "claude-haiku-4-5-20251001"

judge = ClaudeJudge()

# ── 2. LOAD YOUR DAY 1 PIPELINE ──────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from phase1.day1.rag_pipeline import load_documents, chunk_documents, build_vectorstore, query_rag

doc_paths = [
    "phase1/day1/docs/doc1.pdf",
    "phase1/day1/docs/doc2.pdf",
    "phase1/day1/docs/doc3.pdf"
]

docs     = load_documents(doc_paths)
chunks   = chunk_documents(docs)
vs       = build_vectorstore(chunks)

# ── 3. RUN THE PIPELINE FOR Q1 ───────────────────────────
question = "What safety precautions should be followed when mounting the LG TV on a wall?"
expected = "Use a solid wall, hire a professional installer, connect cables before mounting, remove the stand first."

result   = query_rag(vs, question)
actual   = result["answer"]
context  = [doc.page_content for doc in result["context"]]

# ── 4. BUILD THE TEST CASE ───────────────────────────────
test_case = LLMTestCase(
    input=question,
    actual_output=actual,
    expected_output=expected,
    retrieval_context=context
)

# ── 5. RUN BOTH METRICS ──────────────────────────────────
relevancy  = AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=True)
faithfulness = FaithfulnessMetric(threshold=0.7, model=judge, include_reason=True)

relevancy.measure(test_case)
faithfulness.measure(test_case)

# ── 6. PRINT RESULTS ─────────────────────────────────────
print("\n" + "="*60)
print(f"Question : {question}")
print(f"Answer   : {actual[:200]}...")
print("-"*60)
print(f"Answer Relevancy  → Score: {relevancy.score:.2f}  | Pass: {relevancy.success}")
print(f"Reason            : {relevancy.reason}")
print("-"*60)
print(f"Faithfulness      → Score: {faithfulness.score:.2f}  | Pass: {faithfulness.success}")
print(f"Reason            : {faithfulness.reason}")
print("="*60)