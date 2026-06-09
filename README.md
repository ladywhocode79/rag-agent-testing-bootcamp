# RAG Agent Testing Bootcamp

A comprehensive bootcamp for building, testing, and evaluating Retrieval-Augmented Generation (RAG) pipelines with LangChain, Claude, and DeepEval.

## 🎯 Project Overview

This bootcamp provides hands-on training in building production-ready RAG systems. It covers:
- **Phase 1: Foundation** — Building a complete RAG pipeline from document loading to query evaluation
- **Phase 2+: Advanced Topics** — Fine-tuning, optimization, and enterprise patterns

## 📋 Prerequisites

- Python 3.9+
- `pip` or `pip3`
- An Anthropic API key (for Claude models)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip3 install -r requirements.txt
```

Required packages:
- `langchain`, `langchain-community`, `langchain-text-splitters`, `langchain-core`
- `langchain-anthropic`, `langchain-chroma`, `langchain-huggingface`
- `chromadb`, `sentence-transformers`
- `deepeval` (for RAG evaluation)
- `pytest`, `pytest-html` (for testing)
- `python-dotenv` (for environment management)

### 2. Set Up Environment

Create a `.env` file in the project root:

```bash
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

### 3. Prepare Documents

Place PDF documents in `phase1/day1/docs/`:
- `doc1.pdf` (e.g., LG TV manual)
- `doc2.pdf` (e.g., Samsung Galaxy manual)
- `doc3.pdf` (e.g., Sony headphones manual)

## 📁 Project Structure

```
rag-agent-testing-bootcamp/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (git ignored)
├── .gitignore                    # Git ignore rules
│
├── phase1/
│   ├── day1/
│   │   ├── rag_pipeline.py      # Core RAG pipeline implementation
│   │   ├── docs/                # Input PDF documents
│   │   └── chroma_db/           # Vector store (auto-created)
│   │
│   ├── day2/
│   │   └── test_rag_deepeval.py # DeepEval metrics testing
│   │
│   └── day5/
│       └── test_day5_demo.py    # Comprehensive evaluation suite
│
└── course-book/                  # Course materials
    ├── .gitignore
    └── NOTES.md                 # Course notes
```

## 🔄 RAG Pipeline Architecture

### Phase 1: Day 1 — Core Pipeline

The `rag_pipeline.py` implements a complete RAG system:

1. **Load** — Extract text from PDF documents
2. **Chunk** — Split documents into semantic chunks (800 tokens, 80 overlap)
3. **Embed** — Generate embeddings using HuggingFace (all-MiniLM-L6-v2)
4. **Store** — Index embeddings in ChromaDB
5. **Query** — Retrieve context and generate answers with Claude

#### Key Functions

```python
# Load documents from PDFs
docs = load_documents(["path/to/doc1.pdf", ...])

# Split into chunks
chunks = chunk_documents(docs)

# Build vector store
vectorstore = build_vectorstore(chunks)

# Query with RAG
result = query_rag(vectorstore, "Your question here")
# Returns: {"answer": str, "context": List[Document]}
```

#### Query Modes

**Standard RAG** — Uses vector similarity for retrieval:
```python
result = query_rag(vectorstore, question)
```

**Per-Source Retrieval** — Retrieves from each source independently:
```python
answer, chunks = query_rag_per_source(vectorstore, question, doc_paths)
```

## 🧪 Testing & Evaluation

### Phase 1: Day 2 — Basic Evaluation

Run the basic evaluation:

```bash
python3 phase1/day2/test_rag_deepeval.py
```

Metrics evaluated:
- **Answer Relevancy** — Does the answer address the question?
- **Faithfulness** — Is the answer grounded in the retrieved context?

### Phase 1: Day 5 — Comprehensive Evaluation

Run the full test suite with pytest:

```bash
# Run all tests
pytest phase1/day5/test_day5_demo.py -v

# Generate HTML report
pytest phase1/day5/test_day5_demo.py --html=report.html --self-contained-html

# Run with detailed output
pytest phase1/day5/test_day5_demo.py -v -s
```

#### Evaluated Metrics

- **AnswerRelevancy** (threshold: 0.7) — Relevance to the question
- **Faithfulness** (threshold: 0.7) — Grounding in context
- **Hallucination** (threshold: 0.5) — Absence of fabricated facts
- **ContextualRecall** (threshold: 0.7) — Coverage of expected information

#### Test Data

The test suite evaluates 4 questions across 3 device manuals:
- Q1: LG TV wall mounting safety
- Q3: Sony headphones cleaning guidelines
- Q4: LG TV auto-off troubleshooting
- Q5: Battery safety warnings (cross-document)

## 🔧 Configuration

### Vector Store

Edit `phase1/day1/rag_pipeline.py`:

```python
# Chunk size and overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # tokens per chunk
    chunk_overlap=80,      # token overlap
    separators=["\n\n", "\n", ".", " "]
)

# Retrieval parameters
retriever = vectorstore.as_retriever(search_kwargs={"k": 9})  # top 9 chunks
```

### LLM Settings

```python
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",  # or claude-sonnet-4-6
    temperature=0,                       # deterministic
    max_tokens=1024                      # response length
)
```

### Evaluation Thresholds

Edit `phase1/day5/test_day5_demo.py`:

```python
metric = AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=True)
```

## 📊 Understanding Results

### Test Output

```
[Q1] AnswerRelevancy: 0.95 — The score is 0.95 because the actual output 
contains no irrelevant statements and directly addresses the input question.
```

### HTML Report

Generate and view detailed results:

```bash
pytest phase1/day5/test_day5_demo.py --html=report.html --self-contained-html
open report.html
```

The HTML report includes:
- ✓ PASS/FAIL status for each metric
- Detailed reasoning from Claude
- Scores and thresholds
- Full question/answer text

## 🛠️ Troubleshooting

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'langchain'`

**Solution:** Install dependencies
```bash
pip3 install -r requirements.txt
```

### API Key Issues

**Error:** `AuthenticationError: Invalid API key`

**Solution:** Check your `.env` file
```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

### No Documents

**Error:** `FileNotFoundError: phase1/day1/docs/doc1.pdf not found`

**Solution:** Add PDF documents to `phase1/day1/docs/`

### Vector Store Issues

**Error:** `chromadb.errors.InvalidCollectionException`

**Solution:** Delete the vector store and rebuild
```bash
rm -rf phase1/day1/chroma_db
python3 phase1/day1/rag_pipeline.py
```

## 📚 Learning Path

1. **Day 1** → Run `rag_pipeline.py` to understand the pipeline
2. **Day 2** → Run `test_rag_deepeval.py` for basic evaluation
3. **Day 5** → Run `test_day5_demo.py` for comprehensive testing

## 🔑 Key Concepts

- **RAG** — Retrieval-Augmented Generation combines retrieval and generation
- **Embeddings** — Vector representations of text for semantic search
- **Vector Store** — Database for efficient similarity search
- **Context Window** — Retrieved documents fed to the LLM
- **Evaluation Metrics** — Quantify RAG quality (relevancy, faithfulness, etc.)

## 📖 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Claude API](https://docs.anthropic.com/)
- [DeepEval Documentation](https://github.com/confident-ai/deepeval)
- [ChromaDB](https://www.trychroma.com/)

## 📝 Notes

- See `course-book/NOTES.md` for detailed course notes
- All sensitive data (.env files) are ignored in `.gitignore`
- Vector stores and logs are not tracked in git

## 🤝 Contributing

Improvements and suggestions welcome! Test changes thoroughly before committing.

## 📄 License

This bootcamp is for educational purposes.
