
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser


# 1. App setup
st.set_page_config(page_title="Dolphin AI Chatbot")
st.title("🐬 Dolphin AI Chatbot")
st.caption("Ask questions about dolphins using your PDFs and web search.")

base_path = Path(__file__).resolve().parent
data = base_path / "data"
db = base_path / "DB"

load_dotenv(base_path / ".env", override=True)

api_key = os.getenv("GROQ_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
web_domains = [
    domain.strip()
    for domain in os.getenv(
        "WEB_SEARCH_DOMAIN", "wikipedia.org"
    ).split(",")
    if domain.strip()
]

if not api_key:
    st.error("GROQ_API_KEY is missing from your .env file.")
    st.stop()


# 2. Load PDFs and create retrievers
@st.cache_resource
def build_retriever():
    pdf_docs = []

    for filepath in data.glob("*.pdf"):
        loader = PyPDFLoader(str(filepath))
        pdf_docs.extend(loader.load())

    if not pdf_docs:
        raise ValueError("No PDF files found in the data folder.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(pdf_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db.mkdir(exist_ok=True)

    vectorstore = Chroma(
        collection_name="dolphin_books",
        persist_directory=str(db),
        embedding_function=embeddings
    )

    # Add documents only if the collection is empty
    if vectorstore._collection.count() == 0:
        vectorstore.add_documents(chunks)

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 12}
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    retriever = EnsembleRetriever(
        retrievers=[
            vector_retriever,
            mmr_retriever,
            bm25_retriever
        ],
        weights=[0.4, 0.4, 0.2]
    )

    return retriever


try:
    retriever = build_retriever()
except ValueError as e:
    st.error(str(e))
    st.stop()


# 3. Configure Groq
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0,
    reasoning_effort="medium"
)


# 4. Conversational query rewriting
rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Rewrite the user's latest question as a standalone search query.
        Use the conversation history to understand references such as
        "it", "they", "that", or "explain more".
        Keep the original meaning.
        Return only the search query. Do not answer it.
        """
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

rewrite_chain = rewrite_prompt | llm | StrOutputParser()


# 5. Answer prompt
answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a dolphin research assistant.

        Answer using the supplied PDF and web context.
        Give a direct, clear answer. Use simple language.
        Give more detail when the user asks for an explanation.

        Use both sources when relevant.
        Do not invent facts or citations.
        If the sources do not provide enough information,
        say so clearly.

        Cite supporting sources using their source labels
        provided in the context.

        Context:
        {context}
        """
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chain = answer_prompt | llm | StrOutputParser()


# 6. Optional web search
web_search = None

web_search = None

if tavily_key:
    web_search = TavilySearch(
        max_results=5,
        include_domains=web_domains,
        search_depth="basic"
    )


# 7. Chat history stored for this session
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    if source.get("url"):
                        st.markdown(
                            f"- [{source['label']}]({source['url']})"
                        )
                    else:
                        st.markdown(f"- {source['label']}")


# 8. User input
use_web = st.checkbox(
    "Search the web as well",
    value=bool(tavily_key)
)

if use_web and not tavily_key:
    st.warning("Add TAVILY_API_KEY to your .env file to enable web search.")

question = st.chat_input("Ask something about dolphins...")

if question:
    st.chat_message("user").markdown(question)

    history = [
        HumanMessage(content=m["content"])
        if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in st.session_state.messages
    ]

    # Keep recent turns for context
    history = history[-10:]

    # Rewrite follow-up questions for retrieval
    search_query = rewrite_chain.invoke({
        "question": question,
        "chat_history": history
    }).strip()

    if not search_query:
        search_query = question

    # Retrieve PDF passages
    docs = retriever.invoke(search_query)

    context_parts = []
    sources = []
    seen_sources = set()

    for doc in docs:
        filename = Path(
            doc.metadata.get("source", "Unknown file")
        ).name

        page = doc.metadata.get("page")
        page_number = page + 1 if page is not None else None

        label = (
            f"PDF: {filename}, page {page_number}"
            if page_number is not None
            else f"PDF: {filename}"
        )

        context_parts.append(
            f"[{label}]\n{doc.page_content}"
        )

        if label not in seen_sources:
            sources.append({"label": label})
            seen_sources.add(label)

    # Search the specified website
    if use_web and web_search:
        results = web_search.invoke({"query": search_query})

        for result in results.get("results", []):
            title = result.get("title", "Web result")
            url = result.get("url", "")
            content = result.get("content", "")

            if not content:
                continue

            label = f"Web: {title}"

            context_parts.append(
                f"[{label} | URL: {url}]\n{content}"
            )

            if url and url not in seen_sources:
                sources.append({
                    "label": label,
                    "url": url
                })
                seen_sources.add(url)

    context = "\n\n".join(context_parts)

    # Generate the answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chain.invoke({
                "question": question,
                "context": context,
                "chat_history": history
            })

        st.markdown(answer)

        if sources:
            with st.expander("Sources"):
                for source in sources:
                    if source.get("url"):
                        st.markdown(
                            f"- [{source['label']}]({source['url']})"
                        )
                    else:
                        st.markdown(f"- {source['label']}")

    # Save this turn in session memory
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
