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
RAG, 
chunk, 
embedding, 
vector store, 
retriever, 
context window, 
hallucination (in RAG context)

