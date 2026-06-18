<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/LangChain-Framework-green?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Groq-Llama%203.3%2070B-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Cohere-Reranking-purple?style=for-the-badge"/>
<img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-red?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Streamlit-Deployed-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"/>

# 🔍 QueryMind — Production-Grade RAG Evaluation Platform

### Upload any PDF. Ask anything. Get grounded answers — with every response scored for faithfulness and relevance in real time.

**[🚀 Live Demo]([your-streamlit-url-here](https://niranjana-querymind.streamlit.app/))** · **[📽️ Demo Video](your-video-link-here)**

</div>

---

## 🧠 What Is This?

Most RAG projects stop at "it answers questions."

**QueryMind goes further** — it builds the evaluation and observability layer that production AI systems actually need. Every query is automatically scored across three dimensions: faithfulness, answer relevance, and context precision. Every score is logged, visualized, and tracked over time on a live dashboard.

This is the difference between a notebook demo and an engineering system.

---

## ✨ Features

- 📄 **Multi-PDF Upload** — drag and drop any PDF, any domain, multiple files at once
- ✂️ **Intelligent Chunking** — recursive character splitting with configurable overlap
- 🧠 **Local Embeddings** — `all-MiniLM-L6-v2` runs entirely on-device, zero cost
- 🔎 **Semantic Retrieval** — ChromaDB vector similarity search across all ingested chunks
- ⚡ **Cohere Reranking** — top-5 retrieved chunks reranked to top-3 for higher precision
- 💬 **Grounded LLM Answers** — Llama 3.3 70B via Groq API, constrained to document context
- 📊 **Real-time Evaluation Dashboard** — faithfulness, relevance, and precision scored per query
- 🗂️ **Query Log** — every question and answer persisted to SQLite with timestamps
- 🎨 **Premium UI** — glassmorphism design inspired by Vercel, Linear, and Stripe

---

## 🏗️ Architecture

```
User uploads PDF
      │
      ▼
PyPDFLoader → RecursiveCharacterTextSplitter (512 tokens, 50 overlap)
      │
      ▼
HuggingFace Embeddings (all-MiniLM-L6-v2) → ChromaDB (in-memory)
      │
      ▼
User asks question → Similarity Search (top-5 chunks)
      │
      ▼
Cohere Rerank API → top-3 most relevant chunks
      │
      ▼
Groq API (Llama 3.3 70B) → Grounded answer
      │
      ▼
RAGAS-inspired scoring → SQLite log → Live dashboard
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Llama 3.3 70B via Groq | Answer generation |
| **Embeddings** | all-MiniLM-L6-v2 (HuggingFace) | Local vector encoding |
| **Vector Store** | ChromaDB | Semantic similarity search |
| **Reranking** | Cohere Rerank v3 | Precision improvement |
| **Orchestration** | LangChain | RAG pipeline management |
| **Evaluation** | RAGAS-inspired metrics | Answer quality scoring |
| **Database** | SQLite + SQLAlchemy | Query log persistence |
| **Frontend** | Streamlit + Plotly | UI and dashboards |
| **Deployment** | Streamlit Community Cloud | Free hosting |

---

## 📁 Project Structure

```
rag-eval-project/
├── app.py                  # Main Streamlit app (UI + pipeline)
├── src/
│   ├── chain.py            # RAG chain with reranking
│   ├── ingest.py           # Document ingestion pipeline
│   └── logger.py           # SQLite query logger
├── data/
│   └── documents/          # PDF storage (gitignored in production)
├── requirements.txt        # All dependencies
├── .env.example            # Environment variable template
└── README.md
```

---

## ⚙️ Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/Niranjana20055/querymind-rag-evaluation.git
cd querymind-rag-evaluation
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Add your API keys to .env
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🔑 Environment Variables

Create a `.env` file with these keys (all free tier):

```env
GROQ_API_KEY=your_groq_key          # console.groq.com
COHERE_API_KEY=your_cohere_key      # dashboard.cohere.com
LANGCHAIN_API_KEY=your_langsmith_key # smith.langchain.com
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=rag-eval-project
```

---

## 📊 Evaluation Metrics

| Metric | Definition | Target |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? | > 0.85 |
| **Answer Relevance** | Does the answer actually address the question? | > 0.85 |
| **Context Precision** | Were the right chunks retrieved and used? | > 0.80 |

---

## 🔍 Why This Project Stands Out

> Most RAG implementations retrieve and generate. QueryMind retrieves, reranks, generates, **and evaluates** — automatically, on every query, with a live dashboard tracking quality over time.

This reflects how production AI systems are actually built at companies like Google, Anthropic, and Microsoft — where evaluation and observability are as important as the model itself.

---

## 👩‍💻 Author

**Niranjana Vijayaraghavan**  
Final Year B.Tech CSE · VIT Vellore

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/niranjana-vijayaraghavan/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/Niranjana20055)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-purple?style=flat)](https://niranjana-vijayaraghavan-portfolio.lovable.app)

---

<div align="center">
Built with LangChain · Groq · Cohere · ChromaDB · Streamlit
</div>
