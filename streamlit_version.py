# app.py
import streamlit as st
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate

# ── 1. Build / load your LangChain components ──────────────────────────────
# ‑‑‑‑‑‑> make sure these three objects are defined exactly as in your notebook
vector_store = Chroma(
    embedding_function=OllamaEmbeddings(model= "mxbai-embed-large:335m"),
    persist_directory='my_chromadb',
    collection_name='FIrst_Aid_Manual',
)
retriever_3 = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5},
)                                                    # e.g. a VectorStoreRetriever
prompt = PromptTemplate(
    template = """
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      Merge all relevant information from the context to answer the question.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=["context", "question"],
)     # e.g. a PromptTemplate
model = OllamaLLM(model="llama3.2")   # e.g. ChatOpenAI(temperature=0.2)

parser = StrOutputParser()

def context_text(retrieved_docs):
    """Join page_content of retrieved docs into a single context block."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# identical to your original notebook pipeline
parallel_chain = RunnableParallel(
    {
        "context": retriever_3 | RunnableLambda(context_text),
        "question": RunnablePassthrough(),
    }
)
main_chain = parallel_chain | prompt | model | parser
# ───────────────────────────────────────────────────────────────────────────

# ── 2. Streamlit page config ───────────────────────────────────────────────
st.set_page_config(page_title="LangChain × Streamlit Chatbot", page_icon="🤖")
st.title("💬 LangChain Chatbot")
st.caption("Ask me anything…")

# ── 3. Session‑state chat history ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi — how can I help you today?"}
    ]

# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 4. Chat input ↔️ LangChain inference ───────────────────────────────────
if prompt_text := st.chat_input("Type your question here"):
    # ── 4 a. show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # ── 4 b. run LangChain & stream/print the answer
    with st.chat_message("assistant"):
        response_container = st.empty()          # placeholder for streaming
        full_response = ""

        # prefer streaming if your model supports it
        for chunk in main_chain.stream(prompt_text):
            full_response += chunk
            response_container.markdown(full_response + "▌")  # typing cursor

        response_container.markdown(full_response)            # final text

    # save assistant reply to history
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
