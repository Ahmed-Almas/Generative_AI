# All About Dolphins --- RAG Chatbot

A conversational question-answering app about dolphins built with
**LangChain, retrieval-augmented generation (RAG), Chroma, BM25, Groq,
Tavily, and Streamlit**.

The project combines information from two local PDF documents with
optional web search, so users can ask questions about dolphins and
receive answers grounded in the available sources.

## Project status

-   **Phase 1:** `rag.ipynb` --- the notebook used to experiment with
    and test the RAG workflow.
-   **Final app:** `app.py` --- the Streamlit interface for asking
    questions.
-   **Knowledge sources:** two PDF files stored in `data/`.
-   **Vector database:** Chroma data stored in `DB/`.
-   **Environment:** `.venv/` --- the project's Python virtual
    environment.
-   **Dependencies:** `requirements.txt`.

## How it works

1.  The app accepts a question from the user.
2.  It retrieves relevant passages from the local PDF knowledge base
    using Chroma similarity/MMR retrieval and BM25 keyword retrieval.
3.  When enabled and appropriate, it can use Tavily web search. Web
    search is restricted using the configured `WEB_SEARCH_DOMAIN`
    setting.
4.  The retrieved context is passed to the Groq chat model
    (`openai/gpt-oss-120b`) to generate an answer.
5.  Streamlit session state maintains conversational history during the
    app session.

The local documents provide the main RAG knowledge base; web search adds
another source when configured.

## Project structure

``` text
All_About_Dolphins_Using_RAG/
├── .venv/              # Python virtual environment
├── app.py              # Final Streamlit application
├── rag.ipynb           # Phase 1 RAG experiments
├── requirements.txt    # Python dependencies
├── .env                # API keys and configuration (do not commit)
├── data/
│   └── *.pdf           # Two dolphin reference PDFs
└── DB/                 # Chroma vector database
```

The exact filenames of the two PDFs may vary. Keep the PDFs in `data/`
and ensure the app's document-loading paths match their locations.

## Requirements

-   Python version compatible with the packages listed in
    `requirements.txt`
-   Groq API key
-   Tavily API key if web search is used
-   The two PDF reference documents

Install dependencies inside the virtual environment if it has not
already been set up.

### Windows --- activate the existing virtual environment

**PowerShell**

``` powershell
.\.venv\Scripts\Activate.ps1
```

**Command Prompt**

``` bat
.\.venv\Scripts\activate.bat
```

If PowerShell blocks activation, you can run the app using the
environment's Python directly:

``` powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Install dependencies

``` bash
python -m pip install -r requirements.txt
```

### Environment variables

Create or update `.env` in the project root. Use the exact variable
names expected by `app.py`; common settings for this project include:

``` env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
WEB_SEARCH_DOMAIN=example.com
```

Replace the example values with your own credentials and intended domain
restriction. Never publish real API keys or commit `.env` to version
control.

## Run the Streamlit app

From the project root, activate `.venv` and run:

``` bash
streamlit run app.py
```

Or, without activating the environment on Windows:

``` powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will print a local URL in the terminal. Open that URL in your
browser.

## Notebook: Phase 1

Open `rag.ipynb` in Jupyter Notebook or VS Code with the project's
`.venv` selected as the kernel. Use it to explore document loading,
chunking, retrieval, and question answering before running the
integrated Streamlit app.

## Notes and troubleshooting

-   **Missing API key:** check `.env` and confirm that `app.py` loads
    it.
-   **No PDF context found:** verify that both PDFs are in `data/` and
    that the paths in the code match the filenames.
-   **Chroma database issue:** confirm that `DB/` points to the intended
    persisted database and that it was created using a compatible
    embedding configuration.
-   **Web search fails:** verify `TAVILY_API_KEY`, internet access, and
    the `WEB_SEARCH_DOMAIN` setting.
-   **Package errors:** activate `.venv` and install `requirements.txt`
    into that same environment.
-   **Git hygiene:** do not commit `.env`, API keys, or unnecessary
    virtual-environment files.

## Purpose

This project demonstrates a domain-focused RAG application that combines
local PDF retrieval, keyword/vector search, optional web search, and a
Streamlit chat interface to answer questions about dolphins.
