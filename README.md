# BD Demo Project

A local-first, multi-agent document intelligence demo: upload files, ask questions across them, and get grounded answers with citations back to the file, page/slide, section, sheet/cell range or code lines. Built with Streamlit, LangGraph, Ollama, Qdrant, hybrid retrieval with cross-encoder reranking, and pandas-based spreadsheet reasoning.

See [multi_agent_document_chat_plan.md](multi_agent_document_chat_plan.md) for the design.

**Documentation:** [docs/](docs/README.md) contains:
- the [technical documentation](docs/01_TECHNICAL_DOCUMENTATION.md);
- a [setup guide for a new machine](docs/02_SETUP_GUIDE.md);
- a [code walkthrough](docs/03_CODE_WALKTHROUGH.md);
- an [interview demo guide](docs/04_DEMO_GUIDE.md).

## Prerequisites

- Python 3.13+ and [`uv`](https://docs.astral.sh/uv/)
- **Windows only:** [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe). PyTorch (embeddings, reranker) and PyMuPDF (PDF parsing) will not load without it.
- Recommended: [Ollama](https://ollama.com/) with a model pulled (`ollama pull gemma4:e4b`) for LLM answers, routing, verification and image understanding
- Optional: Tesseract OCR, for text in images and scanned PDF pages
- Optional: Docker, only if you want a Qdrant server instead of the built-in in-process Qdrant

Everything degrades gracefully: without Ollama the agents use deterministic fallbacks, and without embeddings retrieval uses BM25 and keyword scoring. The sidebar's **System status** panel shows what is active and why anything is unavailable.

## Setup

```powershell
uv sync --all-extras
```

This installs the runtime dependencies, the `dev` group (pytest) and the `eval` extra from `uv.lock`. The `eval` extra (ragas) is currently unused: RAGAS 0.4.3 doesn't import alongside `langchain-community` 0.4, and replacing it is plan item 1.2.

## Configuration

Settings come from a `.env` file in the project root (optional; defaults live in [app/config.py](app/config.py)). Copy [.env.example](.env.example) to start. The most useful keys:

| Setting | Default | Meaning |
|---|---|---|
| `ENABLE_LLM` | `true` | Use Ollama when reachable |
| `OLLAMA_LLM_MODEL` | `gemma4:e4b` | Chat model (`LLM_MODEL` env var overrides) |
| `OLLAMA_VISION_MODEL` | *(empty)* | Multimodal model for images; empty reuses the chat model |
| `QDRANT_MODE` | `memory` | `memory` (in-process, session-only), `embedded` (on disk at `QDRANT_PATH`), `server` (`QDRANT_URL`) |
| `RERANKER` | `cross-encoder` | `cross-encoder` (`RERANKER_MODEL`) or `lexical` |
| `HYBRID_*_WEIGHT` | `0.35 / 0.35 / 0.30` | Dense / BM25 / keyword weights |
| `MAX_REGENERATIONS` | `1` | Rewrites allowed when verification finds unsupported claims |
| `ALLOW_GENERAL_ANSWERS` | `true` | Answer greetings and general-knowledge questions without sources |
| `ENABLE_LANGSMITH` | `true` | Trace to LangSmith when `LANGCHAIN_API_KEY` is set |
| `LOG_LEVEL` / `LOG_FORMAT` / `LOG_FILE` | `INFO` / `text` / *(none)* | Application logging; `json` gives one object per line, `LOG_FILE` adds a rotating file |
| `LOG_CONTENT` | `false` | Include questions, answers and document text in log lines |
| `LANGSMITH_HIDE_CONTENT` | `false` | Hide questions, document text and answers from LangSmith traces (step metadata is kept) |
| `BACKGROUND_INGESTION` / `INGEST_WORKERS` | `true` / `2` | Ingest uploads on a background thread pool |
| `VISION_AT_INGESTION` | `false` | Describe images with the vision model at upload instead of on the first question about them |

LangSmith is the only component that sends data off the machine; set `ENABLE_LANGSMITH=false` for a fully local run, or keep tracing but set `LANGSMITH_HIDE_CONTENT=true` so traces carry only routes, counts, scores and timings.

## Run the app

```powershell
ollama serve                        # in another terminal, if not already running
uv run streamlit run app/main.py
```

1. Upload files in the sidebar: `.pdf`, `.docx`, `.pptx`, `.txt`, `.md`, `.csv`, `.xlsx`, `.png`, `.jpg`, `.jpeg`, `.py` (up to 50 files, 50 MB each). Each file is parsed, chunked and indexed once.
2. Ask questions. A progress panel shows each step live (routing, retrieval, calculation, writing, claim checking) while the answer streams in token by token with citations rendered as they arrive. Afterwards you get:
   - a verification line (share of claims supported by the evidence),
   - **Sources**: the evidence used, with citations and retrieval/rerank scores,
   - **Claim check**: each claim, whether it is supported and why,
   - **Agent trace**: route, router reasoning, whether the LLM or a fallback wrote the answer, and per-step timings.

Follow-up questions work: "What drove it?" or "and in Q3?" is rewritten into a standalone question using the conversation before searching (the agent trace shows how it was interpreted). The app keeps the last few turns plus, in longer chats, a short summary of earlier turns written by the LLM; nothing is kept after the session ends.

You can also ask general questions or just say hello: greetings, questions about the assistant and general-knowledge questions are answered directly and labelled *General answer — not based on your uploaded files*, with no sources or claim check. Questions about your files never fall back to model knowledge; if the files don't contain the answer, the assistant says so.

Token streaming needs Ollama; without it, answers come from the deterministic fallbacks and appear once complete (the progress panel still updates live).

Uploads are ingested in the background: the sidebar shows each file as queued, processing or ready with a progress bar, and you can ask questions about the files that are ready while the rest finish. Removing a file cancels its ingestion. Uploads live in a per-session temp folder and are deleted (along with their vectors) when removed from the uploader.

## How it works

```text
Upload → parse + chunk (once) → index chunks in Qdrant (session-tagged)
Question → router → parallel agents → synthesis → verification → (regenerate)? → finalize
```

- **Ingestion.** PDFs are parsed in reading order with font-based heading detection (scanned pages go through OCR); DOCX by heading styles with tables kept whole; PPTX one section per slide with tables and notes; Markdown by `#` headings; Python into a module overview plus one chunk per function/class with line numbers; XLSX one table per sheet with its cell range. Images get OCR text at upload; with Ollama available, the multimodal model describes each image the first time a question needs it and the description is cached for later questions (`VISION_AT_INGESTION=true` describes at upload instead).
- **Retrieval.** Dense search (BGE-small in Qdrant, filtered by session and document) is blended with BM25 and keyword overlap, then the top candidates are reranked by a local cross-encoder.
- **Routing.** Keyword rules pick retrieval, data, vision, code, general, or several agents at once; the LLM classifies questions no rule covers (or that only match generic words like "explain"). Agents whose file type isn't uploaded are dropped. The general agent is chosen by rules only for greetings/meta questions or when no files are uploaded; otherwise only the LLM can pick it.
- **Agents run in parallel** (LangGraph fan-out). The data agent works as follows:
  - The LLM chooses and validates a pandas plan: aggregates, top-N, % change between two values, filters (including above/below), and grouped tables. Pandas executes it deterministically.
  - Spreadsheets with title rows above the header are read correctly, and the answer cites the exact sheet range.
  - The rows used are shown as a table under the answer.
  - If a question asks about a year or quarter the data doesn't contain, the agent says so instead of guessing.

  The other agents: the vision agent asks the multimodal model about the relevant images; the code agent retrieves symbols plus their module overview.
- **Synthesis.** The LLM answers only from numbered evidence and cites `[n]`; citations are then rebuilt from evidence metadata, never written by the model.
- **Verification.** The answer is split into claims; each claim's citations are validated, numbers are checked against the evidence, and the LLM judges support. Unsupported answers are regenerated once with feedback; if claims still fail, the answer keeps what is verified and lists what couldn't be verified, or abstains.

## Evaluation

```powershell
uv run python -m app.evaluation.runner            # writes evals/results.json
uv run python -m app.evaluation.runner --sweep    # also grid-searches the hybrid weights
```

The runner generates a sample corpus (report, policy, vendor notes, DOCX notes, PPTX deck, two XLSX workbooks, a Python module, a diagram and, when PyMuPDF loads, a PDF) and runs the 39 questions in [evals/questions.json](evals/questions.json) through the full graph. Results are reported overall and **per category**: retrieval, data, code, vision, multi-agent, conflicting sources, prompt injection, follow-ups (with chat history), general questions and abstention.

Metrics: router accuracy, Recall@K, MRR, answer correctness, citation precision/recall, groundedness, abstention accuracy, general-answer correctness, conflict surfacing, injection resistance and latency. Questions that need a capability the current run lacks (`requires`: `llm`, `pdf`, `vision`) are reported as skipped rather than failed. Some questions deliberately target known gaps (their `note` says which plan item fixes them), so a few failures on the deterministic stack are expected. The runner measures whatever stack is active, so compare runs with and without Ollama/embeddings. Add `--log-level INFO` to see every step of every question.

## Optional services

**Qdrant server** — `docker run -p 6333:6333 qdrant/qdrant`, then set `QDRANT_MODE=server`.

**OCR** — install Tesseract (on Windows, the official build) and check `tesseract --version` in a new terminal.

## Tests

```powershell
uv run python -m pytest -q                                   # all offline tests
uv run python -m pytest tests/test_llm_agents.py -q          # one file
uv run python -m pytest tests/test_router.py::test_name -q   # one test
uv run python -m pytest -m integration                       # integration tests against your local Ollama
```

Integration tests ([tests/test_integration_ollama.py](tests/test_integration_ollama.py)) are excluded from the default run. They exercise the real prompts (grounded answers with citations, abstention, general answers, data answers, regeneration, streaming, follow-up rewriting, routing, the claim judge and vision) and skip with a reason if Ollama isn't running or the model isn't pulled.

[tests/conftest.py](tests/conftest.py) pins the environment (LLM, Qdrant and LangSmith off, lexical reranker), so tests never depend on your `.env`, Ollama or model downloads; LLM, embedding and cross-encoder behaviour is tested with fakes from [tests/fakes.py](tests/fakes.py). PDF tests skip automatically if PyMuPDF cannot load.

## Logs

The app logs to the terminal where Streamlit runs:
- **Startup:** which components are available and why not.
- **Documents:** each document loaded, with its chunk, page and section counts.
- **Questions:** each step of each question with timings.

Every line carries `[session=… request=…]`, so grepping a request ID shows the whole path of one question. Set `LOG_LEVEL=DEBUG` to also get a **Logs** panel in the sidebar, and `LOG_FILE=logs/app.log` to keep a rotating file. Questions, answers and document text are left out of the logs unless `LOG_CONTENT=true`.

## Troubleshooting

**`DLL load failed` / "Microsoft Visual C++ Redistributable is not installed"** — install the [VC++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) and restart the terminal. Until then PDFs show as `failed`, and dense retrieval and the cross-encoder are unavailable (see System status).

**`ModuleNotFoundError: No module named 'app'` when starting Streamlit** — the project isn't installed in the virtual environment. Run `uv sync --all-extras` from the project root (it installs `app` in editable mode) and start the app with `uv run streamlit run app/main.py`.

**System status says the LLM is unavailable** — start `ollama serve` and `ollama pull <model>` for the configured model. The app re-checks every 30 seconds.

**A file shows status `failed`, `empty` or `unsupported`** — `failed`: the parser raised (corrupt file or missing native library); `empty`: no text could be extracted; `unsupported`: the extension isn't handled. These files are excluded from answers.

**Image questions abstain** — images need Tesseract (OCR) or an Ollama multimodal model to have any content to answer from.
