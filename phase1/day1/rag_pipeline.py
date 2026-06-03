import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ── 1. LOAD ──────────────────────────────────────────────
def load_documents(doc_paths: list[str]):
    docs = []
    for path in doc_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    print(f"Loaded {len(docs)} pages from {len(doc_paths)} documents")
    return docs

# ── 2. CHUNK ─────────────────────────────────────────────
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    return chunks

# ── 3. EMBED + STORE (local, no API key needed) ───────────
def build_vectorstore(chunks):
    # Using a free local embedding model — no OpenAI key needed
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./phase1/day1/chroma_db"
    )
    print(f"Stored {vectorstore._collection.count()} vectors in ChromaDB")
    return vectorstore

# ── 4. QUERY ──────────────────────────────────────────────
def query_rag(vectorstore, question: str):
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",   # fast + cheap for Day 1
        temperature=0,
        max_tokens=1024
    )

    system_prompt = """You are a helpful assistant answering questions based on provided documents.
    Use only the retrieved context below to answer accurately.
    If the context doesn't contain the answer, say "I cannot find this in the provided documents."

    Context:
    {context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 9})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)
    context_docs = retriever.invoke(question)

    return {"answer": answer, "context": context_docs}


def query_rag_per_source(vectorstore, question: str, doc_paths: list):
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0,
        max_tokens=1024
    )

    # Retrieve 3 chunks per source document independently
    all_chunks = []
    for path in doc_paths:
        source_name = os.path.basename(path)
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 3,
                "filter": {"source": path}  # filter by source document
            }
        )
        chunks = retriever.invoke(question)
        all_chunks.extend(chunks)
        print(f"  Retrieved {len(chunks)} chunks from {source_name}")

    # Build context from all collected chunks
    context = "\n\n".join([c.page_content for c in all_chunks])

    system_prompt = """You are a helpful assistant answering questions based on provided documents.
Use only the retrieved context below to answer accurately.
If the context doesn't contain the answer, say "I cannot find this in the provided documents."

Context:
{context}"""

    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    chain = prompt | llm
    result = chain.invoke({
        "context": context,
        "input": question
    })

    return result.content, all_chunks
# ── 5. MAIN ───────────────────────────────────────────────
if __name__ == "__main__":
    doc_paths = [
        "phase1/day1/docs/doc1.pdf",
        "phase1/day1/docs/doc2.pdf",
        "phase1/day1/docs/doc3.pdf"
    ]

    docs = load_documents(doc_paths)
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks)

    # Replace these with REAL questions about YOUR documents
    questions = [
    "What safety precautions should be followed when mounting the LG TV on a wall?",
    "What wireless connectivity standards does the Samsung Galaxy S24 support?",
    "How should I clean the Sony headphones and what materials should I avoid?",
    "What should I do if the LG TV turns off suddenly and what are the possible causes?",
    "What are the battery safety warnings mentioned across any of the three device manuals?"
]
    
    print("\n" + "="*60)
    for i, q in enumerate(questions, 1):
        if i == 5:
            # Use per-source retrieval for cross-document question
            answer, source_docs = query_rag_per_source(vectorstore, q, doc_paths)
            print(f"\nQ{i}: {q}")
            print(f"Answer: {answer}")
            print(f"Sources: {len(source_docs)} chunks retrieved (3 per document)")
            for j, doc in enumerate(source_docs):
                print(f"  Chunk {j+1} (page {doc.metadata.get('page','?')}): "
                      f"...{doc.page_content[:150]}...")
        else:
            result = query_rag(vectorstore, q)
            print(f"\nQ{i}: {q}")
            print(f"Answer: {result['answer']}")
            print(f"Sources: {len(result['context'])} chunks retrieved")
            for j, doc in enumerate(result['context']):
                print(f"  Chunk {j+1} (page {doc.metadata.get('page','?')}): "
                    f"...{doc.page_content[:150]}...")
        print("-"*60)