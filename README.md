# 🩹 **First‑Aid Chatbot**  
_Retrieval‑Augmented Guidance from a Trusted Medical Manual_

---

## 🧭 Project Vision

Create an **interactive chatbot** that gives **fast, accurate, and context‑grounded first‑aid instructions** for any emergency scenario a user describes (burns, snake‑bites, fainting, etc.).  

The assistant should:

1. **Retrieve** facts _only_ from trusted medical sources.  
2. **Answer** with concise, step‑by‑step guidance.  
3. **Cite** the page / section used.  
4. **Remember** the conversation to handle follow‑up questions.  
5. **Run fully offline** using local LLMs (Ollama).

---

## 🌟 Current Features (v0.2)

| Layer | Technology | Notes |
|-------|------------|-------|
| **LLM** | `llama3.2` via **Ollama** (local) | Streaming responses |
| **Embeddings** | `mxbai‑embed‑large:335m` | High‑quality vectors |
| **Vector Store** | **ChromaDB** (persisted) | Stores >5 K chunks |
| **Document Source** | _Indian First Aid Manual (2016, 7th ed.)_ | Loaded via `PyPDFLoader` |
| **Chunking** | `RecursiveCharacterTextSplitter` (1 400 chars, 10 % overlap) | |
| **Retrievers** | `similarity` • `MMR` • _Multi‑Query_ • _Contextual Compression_ | Empirically best: `MMR` (k=3, λ=0.5) |
| **Prompt** | “Answer ONLY from context… merge relevant info…” | Hallucination guardrail |
| **LangChain Pipeline** | `RunnableParallel` → `PromptTemplate` → `OllamaLLM` → `StrOutputParser` | LCEL |
| **Frontend** | **Streamlit** chat UI | Typing cursor, session history |

---

## 🚧 Known Limitations

1. **No conversational memory** – bot forgets previous turns.  
2. **Cold‑start latency** (~6 s first token) – need caching & model streaming.  
3. **Retrieval quality** is “okay” but not perfect on edge cases.  
4. **Single data source** – manual only; lacks WHO / Red Cross updates.

---
