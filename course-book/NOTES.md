## Day 1 — RAG fundamentals
Q1. How many chunks did each document produce? Why might that number matter for retrieval quality?
On first run each in total all 3 documents created 1115 chunks and Loaded 369 pages.But for Q4 , shows a chunking boundary failure — a troubleshooting table split mid-row lost context. Hence Larger chunk_size was added as 800 as suggested and for Q5 shows that k=3 is too small for cross-document queries. Hence updated it to k=9 and rerun — so that other 2 documents are also considered.Post fix Q4 answer improved but Q5 , all 9 chunks were from one document instead of 3 which was due to real RAG limitation called source imbalance — one document dominates retrieval on certain topics. Fix was added for Q5 as special condition/post fix and clean up  chunk size became Created 751 chunks.
Chunk size matters for retrieval quality.Smaller chunks = more precise but risk losing context (Q4's problem). Larger chunks = more context but retrieval can become less precise because the chunk contains too many topics. Speed is a minor secondary effect — don't lead with it.

Q2. For each of your 5 queries, did the retrieved chunks actually contain the answer? If not, which failure point (F1–F5) do you think caused it?
We asked about S24 but the PDF was S23.
Wrong document — but this is interesting
The answer says "I cannot find info on S24 — these documents mention S23." That is the correct honest answer given what was retrieved. But look at Chunk 2 — page 2 literally shows a table of contents entry for "Wi-Fi", "Bluetooth", "NFC". The info is in the document, but the retriever pulled regulatory pages (page 175) and table of contents text instead of the actual connectivity section.
Root cause: Your PDF is likely a Samsung S23 manual, not S24. The model correctly refused to hallucinate — that's good behaviour. But it tells you: check which PDF you dropped in. If it's actually S23, update your question to match. Mismatched questions are a common Day 1 mistake.

Q3. Look at your chunks. Find one example of a bad chunk — one that was split in a way that destroyed context. Paste it in your notes.
Chunk 1 from Q4's first run is a perfect example — it was a safety warning page that appeared instead of the troubleshooting table. Write: "Chunk from page 3 (safety warnings) appeared when asking about TV shutdown troubleshooting — the splitter broke the troubleshooting table mid-row and mixed it with unrelated safety content."

Q4. What was your chunk size (500) and overlap (50)? Explain in one sentence what "overlap" does and why it exists.
 751 chunks.script has chunk_overlap=80. Overlap means the last 80 characters of chunk N are repeated at the start of chunk N+1. You won't see it visually in the output — you'd have to print two consecutive chunks side by side to spot it.Overlap means repetition of sentence in multiple chunks to help LLM to reterive data with correct context.


## Three things worth writing down right now:

Q2 reveals a question-document mismatch — always verify your question matches what's actually in the PDF, not what you think is in it.
Q4 shows a chunking boundary failure — a troubleshooting table split mid-row lost context. Larger chunk_size (try 800) or semantic chunking would help.
Q5 shows that k=3 is too small for cross-document queries. Try k=9 and rerun — you should start seeing Sony and LG chunks appear alongside Samsung.

## What to write in your notes today
You've now seen all 5 RAG failure modes in a single session:
Failure.                        Where you saw it
Good retrieval                  Q1, Q3
Question-document mismatch      Q2 (S24 vs S23)
Chunking boundary split         Q4 first run, fixed by chunk_size=800
Source imbalance                Q5 first two runs, fixed by per-source retrievalNo relevant content existsQ5 LG chunks — model correctly refused to hallucinate


## Also Add these definitions
As you read, watch, and build — write your own definitions for: 
RAG: Reterival Augemented Generation is a process/setup wherein with LLM's are provided supporting lates documents / references , so that agents can relevant and updated answers .

chunk : Documents/Data are divided into small chunks before saving into vector db for quick rterival by agents."Too small and you lose context (a sentence without its paragraph). Too large and retrieval becomes imprecise (too many topics in one chunk)." 

embedding : Embedding converts text into a list of numbers (a vector) where similar meanings produce similar numbers, enabling similarity search instead of keyword matching.

vector store : A database that stores embeddings and retrieves the most similar ones to a given query using similarity scoring (e.g. cosine similarity).
retriever: Takes the user's question, embeds it, searches the vector store for the closest chunks, and passes those chunks as context to the LLM.
context window:The maximum amount of text (question + retrieved chunks + answer) an LLM can process in a single call. If retrieved chunks are too large or too many, they won't fit.

hallucination (in RAG context): When the LLM generates an answer not supported by the retrieved chunks — either because no relevant chunk was found, or because the LLM ignored the context and invented a plausible-sounding but incorrect answer.


## Day 2 — DeepEval basics: AnswerRelevancy + Faithfulness

Q1. What AnswerRelevancy measures in one sentence
Measures how directly and completely the answer addresses the question asked.

Q2. What Faithfulness measures in one sentence
Measures whether the answer is grounded in the retrieved chunks, with no claims added beyond what the context contains.

Q3. Why 1.00 on both doesn't mean the whole pipeline is perfect (hint: think about Q2 and Q5 from Day 1)
High scores on AnswerRelevancy and Faithfulness do not mean your pipeline is correct. They only mean the answer was on-topic and grounded in whatever was retrieved. If the wrong document was retrieved, or a document was missing entirely, the metrics still pass — and you'd never know.
This is why retrieval metrics (Contextual Precision and Recall) exist 

## Some more questions from Day2
Q1. What is an LLMTestCase and what 4 things does it hold?
It's a container — exactly like a test case in any QA framework. It holds 4 things:
input            → the question you asked
actual_output    → what your RAG pipeline returned
expected_output  → what a correct answer looks like
retrieval_context → the chunks that were retrieved

Q2. What is the difference between a metric and a threshold?
A metric is the measurement — AnswerRelevancyMetric scores 0.0 to 1.0.
A threshold is your pass/fail line — you set threshold=0.7, so anything below 0.7 fails.
Exactly like this in traditional testing:
response_time = measure()        # metric
assert response_time < 2000      # threshold

Q3. Why does DeepEval use another LLM as the judge instead of checking the answer directly?

Because you can't use == on a sentence. These two answers are both correct but would fail an exact match:
Expected : "Wipe with a soft dry cloth"
Actual   : "Use a dry soft cloth to clean the surface"
So instead you ask Claude: "Does this answer correctly address the question? Score it 0 to 1." That's the entire idea behind LLM-as-judge.

4.so is deepEval is all about LLM as a judege?
Mostly yes, but not entirely. DeepEval has three types of checks:

1. LLM-as-judge (most metrics)
AnswerRelevancy, Faithfulness, Contextual Precision, Hallucination — all of these use an LLM to score the output. This is the core of DeepEval and what you'll use 80% of the time.
2. Non-LLM checks (deterministic)
Some metrics don't need an LLM at all — they check things mechanically:

    Does the output contain a specific keyword?
    Is the output under a certain length?
    Does it match a regex pattern?

These are fast, cheap, and 100% deterministic. Useful for basic guardrails.
3. Statistical similarity
Compares outputs using embedding similarity scores rather than an LLM judge. Faster than LLM-as-judge but less nuanced.

## Day 3 Contextual Precision and Recall.
Contextual Precision — of the chunks retrieved, how many were actually useful for answering the question? If you retrieved 9 chunks and only 2 were relevant, precision is low. This is your retriever being noisy.

Contextual Recall — of all the useful chunks that exist in the documents, how many did the retriever actually find? If 5 relevant chunks exist but you only retrieved 2, recall is low. This is your retriever missing things.

Mapped to what you already saw on Day 1:

Q4 first run — retriever pulled safety warning pages instead of the troubleshooting table → low precision (noisy chunks)
Q5 — retriever missed Sony and LG chunks entirely → low recall (missing relevant content)

## Day notes
Precision = retriever noise problem
Recall = retriever missing problem
High scores on Relevancy and Faithfulness with low Precision means: good answer, bad retriever
You can have perfect Recall and still have low Precision — they are independent

## some observations at Day 3
Your prediction was right — precision is lower for Q4 than Q1. But not as low as expected. Here's why that's interesting.
0.83 is not a failure, but read the reason carefully.
The judge is telling you exactly what happened: the two relevant chunks landed at positions 1 and 3, but an irrelevant safety warning chunk slipped into position 2. Precision penalises that because in an ideal retriever all relevant chunks come first, then all irrelevant ones. One noisy chunk in the middle dropped you from 1.00 to 0.83.
This maps directly to what you saw on Day 1 — that safety warning page from page 3 kept showing up in Q4 results even though it had nothing to do with TV shutdown troubleshooting.

Recall is 1.00, not low.
Here's why. Recall measures whether the retriever found all the relevant chunks that exist. For Q4 the two relevant chunks — the power settings chunk and the auto-off chunk — were both retrieved. All expected content was found. So recall is perfect.
What was low was precision — a noisy chunk slipped in at position 2.
This is a critical distinction to lock in:
Precision and Recall fail for different reasons and point to different fixes:
ProblemMetric that catches itFixRetriever pulls irrelevant chunksLow PrecisionSmaller chunk size, better embeddingsRetriever misses relevant chunksLow RecallLarger k, better chunking strategy
Q4 has a precision problem, not a recall problem. The relevant content was found — it just had noise mixed in.
Q5 from Day 1 was the opposite — recall was the real failure there because Samsung dominated and Sony/LG chunks were never retrieved at all.

## Day 4 Hallucination detection

Hallucination is related but catches something different. Here's the distinction in one line each:
Faithfulness — did the answer use the retrieved context?
Hallucination — did the answer contradict or go beyond the retrieved context?
An answer can be faithful but still hallucinate. Example:
Context   : "Clean with a soft dry cloth."
Answer    : "Clean with a soft dry cloth. You can also use warm water."
Faithful — yes, it used the context. Hallucination — yes, it added "warm water" which was never in the context.
Faithfulness checks inclusion. Hallucination checks invention. You need both.

What HallucinationMetric actually measures:
Score 0.0 = no hallucination = answer aligns with context = PASS
Score 1.0 = hallucination detected = answer contradicts context = FAIL

## Imp notes for Hallucination
HallucinationMetric 0.0 = no hallucination = PASS, 1.0 = contradiction detected = FAIL
The metric detects contradiction between answer and context — it does not decide which one is truthful
In production, retriever quality directly determines how reliable this metric is
This confirms the lesson from earlier:the metric is only as reliable as the context you feed it. In production your context comes from your retriever — so a bad retriever producing wrong chunks will cause good answers to fail this metric.

## Day 5 Pytest integration and test report
Day 5 is just wiring everything you built on Days 2, 3, and 4 into a single proper test file with fixtures, parametrize, and a clean summary at the end.
The only new idea today is fixtures for DeepEval — instead of rebuilding the vectorstore inside every test function, you build it once and share it across all tests. Same principle as a @pytest.fixture you already know from API or UI testing.

scope="session" on the fixtures means the vectorstore is built once for the entire test session, not once per test. Without this, your pipeline rebuilds and re-embeds all PDFs for every single test function — 10 tests would rebuild 10 times. With it, it builds once and all tests share the same instance.
ids=[d["id"] for d in test_data] gives your tests readable names in the output — test_answer_relevancy[Q1] instead of test_answer_relevancy[item0].

What this suite has now proven across 4 days:
Metric.               Purpose.                     Caught real issue?
AnswerRelevancy.      Answer on-topic?             Q5 slightly off (0.86)
Faithfulness.         Answer grounded in context?  Q4 slight drift (0.80)
ContextualPrecision   Noisy chunks?                Q4 noise at position 2 (Day 3)
ContextualRecall.     Missing chunks?              Fixed by per-source retrieval
Hallucination.        Answer contradicts context?  Caught fake context correctly

## Imp notes of Day5

scope="session" fixture builds vectorstore once — critical for performance
16/16 passed but 0.80 and 0.86 scores are warnings worth investigating in production
Passing all tests at threshold 0.7 does not mean the pipeline is perfect

Two things to understand why:
Why Q5 passed Recall: Your expected output for Q5 mentions Samsung charging rules, temperature range, and Sony's red flash. The per-source retrieval fix you built on Day 1 ensured chunks from all 3 documents were retrieved. So the judge found support for all expected sentences in the context — Recall passed.

Why Q4 Faithfulness scored 0.80: This is the most interesting result in the whole run. The judge caught a subtle issue — your pipeline described the auto-off as something in "time settings" but the manual says it triggers automatically after 15 minutes without input. That's a real faithfulness gap — the answer slightly misrepresented the mechanism. It still passed at 0.7 threshold, but it's a genuine finding worth noting.

## Day 6 — Threshold tuning and score analysis

You have been running all tests at threshold 0.7. But how do you know 0.7 is the right threshold for your pipeline?
In traditional testing a pass/fail is binary — either the button works or it doesn't. In LLM testing the line is movable. Setting it too low means bad answers pass. Setting it too high means good answers fail. Day 6 is about understanding where your pipeline actually sits and making a conscious decision about what threshold is appropriate.

Then answer these two questions in your NOTES.md before coming back:

Which question and metric combination is borderline — passes at 0.7 but fails at 0.9? What does that tell you about that part of your pipeline?
Your explanation is correct — it is a ranking/prioritisation problem, not a missing content problem. The relevant chunks were retrieved but noisy chunks appeared between them, pushing the score below 0.9. Well understood.
One thing to add to your notes: this is a retriever quality issue, not an LLM quality issue. The LLM answered correctly — the retriever just didn't rank cleanly enough to hit 0.9.

Q2 — Contextual Recall for Q5
Correct. Not all relevant chunks were retrieved — specifically cross-document content. You already know the root cause: source imbalance. The per-source fix helped but wasn't perfect.

If this were a production healthcare chatbot giving medication instructions, would you set threshold at 0.5, 0.7, or 0.9 — and why?

0.9 is the right answer and your reasoning is correct. In a domain where a wrong answer causes patient harm, you want the bar high. A medication chatbot scoring 0.83 on ContextualPrecision means noisy chunks are influencing the answer — that is unacceptable when the stakes are a dosage instruction.

## Imp notes
Pipeline weakest point: ContextualPrecision — noisy chunk ranking
All other metrics hit 0.9 cleanly — answer quality and faithfulness are strong
Production threshold decision depends on domain risk — healthcare = 0.9, general FAQ = 0.7


