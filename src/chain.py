from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

def rerank_and_answer(question):
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory="data/chroma_db",
        embedding_function=embeddings
    )

    # Step 1 - Retrieve top 5 chunks
    raw_docs = vectorstore.similarity_search(question, k=5)
    docs_text = [doc.page_content for doc in raw_docs]

    # Step 2 - Rerank using Cohere
    reranked = co.rerank(
        query=question,
        documents=docs_text,
        top_n=3,
        model="rerank-english-v3.0"
    )
    top_docs = [raw_docs[r.index] for r in reranked.results]

    # Step 3 - Build context from top reranked docs
    context = "\n\n".join([doc.page_content for doc in top_docs])

    # Step 4 - Send to Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    response = llm.invoke(
        f"""You are a helpful assistant. Use only the context below to answer.
If the answer is not in the context, say exactly:
"I don't know based on the documents provided."
Never make up an answer.

Context: {context}

Question: {question}

Answer:"""
    )

    return response.content, top_docs