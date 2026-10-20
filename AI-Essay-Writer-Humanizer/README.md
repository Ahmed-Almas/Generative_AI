# Essay Writer, Humanizer & AI Detector

A Python project built with **LangGraph, LangChain, Groq, web search,
PDF input, Gradio, and external text-processing APIs**. It takes a
writing request, creates an essay, humanizes the generated text, and
checks the humanized result with an AI-detection service.

## Project status

-   **Phase 1:** `human.ipynb` --- notebook experiments for the early
    workflow, including web search.
-   **Graph workflow:** `graph.py` --- defines the LangGraph workflow
    and is the entry point used by LangGraph Studio.
-   **User interface:** `app.py` --- Gradio interface for submitting a
    writing request and optionally uploading a PDF.
-   **Graph configuration:** `langgraph.json`.
-   **Dependencies:** `requirements.txt`.
-   **Environment:** `.venv/`.
-   **Secrets/configuration:** `.env` (keep private).
-   **Optional reference files:** PDFs can be provided through the
    Gradio upload control; the project may also contain a `documents/`
    folder for local reference files.

## Features

-   Accepts a natural-language request such as:
    `Write an essay about dolphins in 300 words.`
-   Extracts the topic and requested length from the prompt.
-   Searches the web when no PDF is supplied.
-   Accepts an optional PDF as a reference source.
-   Generates an original essay.
-   Sends the essay to the configured humanization service.
-   Checks the humanized text with the configured AI-detection service.
-   Returns the original essay, humanized essay, final word count, and
    detector result.

**Important:** AI detectors can disagree and may misclassify
human-written or AI-assisted text. The detector result is an estimate,
not proof of authorship. Humanization also does not guarantee that a
detector will label text as human-written.

## Workflow overview

``` text
Writing request + optional PDF
              |
              v
       Parse the request
              |
       +------+------+
       |             |
     PDF given     No PDF
       |             |
       v             v
   PDF retrieval   Web search
       |             |
       +------+------+
              |
              v
       Generate essay
              |
              v
        Humanize text
              |
              v
      Check AI detection
              |
              v
 Return essays, word count,
       and detector result
```

Depending on the graph's current routing logic, supplying a PDF may
route the request through PDF retrieval instead of web search. The early
notebook and final graph may not have identical behavior.

## Project structure

``` text
Essay_Humanizer/
├── .venv/              # Python virtual environment
├── graph.py            # LangGraph workflow
├── langgraph.json      # LangGraph Studio configuration
├── app.py              # Gradio web interface
├── human.ipynb          # Phase 1 notebook, including web-search experiments
├── requirements.txt    # Python dependencies
├── .env                # API keys (do not commit)
└── documents/           # Optional local reference PDFs, if used
```

## Requirements

-   A compatible Python version for the packages in `requirements.txt`
-   Groq API key for essay generation
-   Tavily API key if web search is used
-   Credentials or access for the configured humanization and
    AI-detection APIs
-   A PDF file if you want to use PDF-based context

Install and run everything from the project root so relative paths and
configuration are resolved consistently.

## Windows setup

### Activate the existing virtual environment

**PowerShell**

``` powershell
.\.venv\Scripts\Activate.ps1
```

**Command Prompt**

``` bat
.\.venv\Scripts\activate.bat
```

If activation is blocked or inconvenient, run commands using the
environment's Python executable directly:

``` powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Install dependencies

``` bash
python -m pip install -r requirements.txt
```

If the environment is already configured, you can skip installation
unless dependencies have changed.

## Configure environment variables

Create or update `.env` in the project root. Use the **exact variable
names referenced in `graph.py`**. The services used by this workflow may
require keys for Groq and Tavily, plus any credentials required by the
humanization and AI-detection APIs.

Example placeholders:

``` env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Add any other service credentials required by the current code. Do not
commit `.env` or publish API keys.

## Run the Gradio app

From the project root:

``` bash
python app.py
```

Or on Windows without activating `.venv`:

``` powershell
.\.venv\Scripts\python.exe app.py
```

Open the local URL printed by Gradio. Enter a writing request and, if
needed, upload a PDF using the optional PDF control. The app returns the
original essay, humanized essay, final word count, and AI-detector
result.

## Open the graph in LangGraph Studio

`graph.py` defines the graph and `langgraph.json` configures how
LangGraph tooling loads it. Start LangGraph Studio using the workflow
supported by your installed LangGraph CLI and the configuration in this
repository. Run commands from the project root with `.venv` active.

If Studio cannot load the graph, check that: - `langgraph.json` points
to the correct graph module and graph variable. - All packages in
`requirements.txt` are installed in `.venv`. - Required environment
variables are available to the Studio runtime. - `graph.py` can be
imported without errors.

Gradio is the end-user interface; LangGraph Studio is used to inspect
and run the graph workflow.

## Notebook: Phase 1

Open `human.ipynb` in VS Code or Jupyter Notebook and select the
project's `.venv` as the kernel. It represents the earlier experimental
stage and includes web-search work. The integrated graph and Gradio app
are the later application stage.

## PDF input notes

The Gradio app accepts a PDF file and passes its temporary file path to
the graph. PDF extraction depends on the installed loader packages and
the PDF's contents. Text-based PDFs are easier to extract than
scanned/image-only PDFs. If a PDF cannot be read, check the error
message, confirm the file path exists during execution, and verify that
the required PDF loader is installed. Scanned PDFs may require OCR.

## Troubleshooting

-   **`pip` is not recognized:** use `python -m pip ...` or the
    `.venv\Scripts\python.exe` path shown above.
-   **Graph import error:** run commands from the project root and check
    that `graph.py` exports the graph variable expected by `app.py` and
    `langgraph.json`.
-   **PDF extraction returns zero text:** the file may be
    scanned/image-based, the loader may be missing, or the path may be
    incorrect.
-   **Web search fails:** check the Tavily key and network access.
-   **Humanization/detection API fails:** check the API endpoint,
    response format, credentials, and service availability.
-   **Detector results differ from other tools:** detectors use
    different methods and can produce different scores or verdicts.

## Purpose

This project demonstrates an end-to-end LangGraph workflow for essay
generation with optional reference-PDF context or web search, followed
by humanization and AI-detection checking, with Gradio for the
user-facing interface and LangGraph Studio for graph development.
