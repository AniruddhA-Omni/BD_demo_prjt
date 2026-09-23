# Multi-Agent Document Chat — Interview Showcase Plan

## 1. Project Objective

Build a robust, local-first, multi-agent document intelligence application that demonstrates production-oriented AI/ML engineering skills for interviews.

The application should let a user upload up to **50 files**, ask questions across them, perform structured spreadsheet reasoning, understand images/charts/screenshots, explain code, and return **grounded answers with precise citations**.

The goal is **not** to literally reproduce Copilot/ChatGPT feature-for-feature. The goal is to demonstrate the engineering concepts that make those systems strong:

- Multi-format ingestion
- Layout-aware parsing
- Hybrid retrieval + reranking
- Multi-agent orchestration
- Spreadsheet reasoning
- OCR + visual understanding
- Cross-document reasoning
- Session-aware conversation context
- Citation grounding
- Verification / hallucination control
- Evaluation and observability
- Fully local execution with Ollama

---

## 2. Confirmed User Requirements

| Area | Decision |
|---|---|
| Product goal | **Interview showcase project** |
| Functional scope | **All major capabilities**: document chat, cross-document reasoning, spreadsheet reasoning, agentic workflows |
| UI | **Streamlit** |
| Project/dependency management | **uv** |
| Agent orchestration | **LangGraph** |
| LLM provider | **Ollama** |
| LLM model | **Gemma 4 via Ollama**; exact Gemma 4 variant configurable, with a lightweight variant preferred |
| Execution model | **Local application + local inference/data**; LangSmith tracing enabled when credentials are configured |
| Hardware | **RTX 5050 + 32 GB RAM** |
| Initial per-file size | **< 50 MB** |
| Maximum files/session | **50** |
| Retrieval | **Hybrid + reranking** |
| Retrieval complexity | **Keep V1 relatively simple**; avoid excessive retrieval tricks |
| PDF parsing | **Layout-aware** |
| Image understanding | **OCR + visual understanding** |
| OCR | **Tesseract** |
| Spreadsheet reasoning | **Pandas-based**; no DuckDB |
| Code support | **Explain code + generate documentation** |
| Conversation memory | **Session-only** |
| Citations | **File + page/section + exact paragraph/table/cell range where possible** |
| Router | **Hybrid routing**: deterministic rules + LLM classification |
| Observability | **LangSmith; credentials available, use freely for tracing/evaluation** |
| Authentication | **None** |
| Deployment | **Python run first**; Docker only when useful/needed |

---

# 3. High-Level Architecture

```text
                              ┌─────────────────────┐
                              │     Streamlit UI    │
                              │ Upload + Chat + UX  │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │ Session / Workspace │
                              │ File + Chat State   │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  Ingestion Router    │
                              └──────────┬──────────┘
                                         │
             ┌───────────────┬───────────┼───────────┬───────────────┐
             │               │           │           │               │
             ▼               ▼           ▼           ▼               ▼
          PDF/DOCX        PPTX/TXT     CSV/XLSX    Images          Code
             │               │           │           │               │
             ▼               ▼           ▼           ▼               ▼
        Layout-aware      Structure    Pandas     Tesseract +      Code-aware
          parsing         parsing      tables       Vision          parsing
             │               │           │           │               │
             └───────────────┴───────────┼───────────┴───────────────┘
                                         ▼
                              ┌────────────────────┐
                              │ Canonical Document │
                              │ / Evidence Model   │
                              └─────────┬──────────┘
                                        │
                      ┌─────────────────┼────────────────┐
                      ▼                 ▼                ▼
                Vector Index       BM25 Index       Structured Data
                (semantic)         (keyword)        (DataFrames)
                      │                 │                │
                      └─────────────────┼────────────────┘
                                        ▼
                              ┌────────────────────┐
                              │ Hybrid Query Router│
                              │ Rule + LLM         │
                              └─────────┬──────────┘
                                        │
                   ┌────────────────────┼────────────────────┐
                   ▼                    ▼                    ▼
             Retrieval Agent      Data Agent           Vision Agent
             semantic + BM25      pandas reasoning     OCR + image QA
                   │                    │                    │
                   └────────────────────┼────────────────────┘
                                        ▼
                              ┌────────────────────┐
                              │ Evidence Fusion /  │
                              │ Reranking Layer    │
                              └─────────┬──────────┘
                                        ▼
                              ┌────────────────────┐
                              │  Synthesis Agent   │
                              └─────────┬──────────┘
                                        ▼
                              ┌────────────────────┐
                              │ Verification Agent │
                              │ claims + evidence  │
                              └─────────┬──────────┘
                                        ▼
                           Final answer + citations
                           + confidence/evidence UI
```

---

# 4. Supported Formats

## Core V1

### Documents
- PDF
- DOCX
- TXT
- Markdown
- PPTX

### Structured data
- CSV
- XLSX

### Images
- PNG
- JPG/JPEG
- scanned document images
- screenshots
- diagrams/charts

### Code
- Python initially
- extensible to other text/code files later

## Optional Later Extensions

- HTML/web pages
- Audio transcripts
- Video transcripts

These should **not** block the core project.

---

# 5. Recommended Technology Stack

# 5.1 Local Resource Profile

The system is intentionally designed for a **low-resource local workflow**:

- Hardware target: **RTX 5050 + 32 GB RAM**
- LLM: Gemma 4 through Ollama
- Embeddings: **BGE-small**
- Retrieval: hybrid BM25 + vector search
- Reranking: lightweight local cross-encoder
- No DuckDB
- No cloud LLM APIs
- No cloud vector DB or cloud inference
- LangSmith may receive tracing/telemetry when enabled
- No authentication or multi-user infrastructure
- Python/uv first; Docker only when it adds reproducibility value

The design priority is:

```text
local reliability > model size > infrastructure complexity
```

The application, LLM inference, embeddings, retrieval, OCR, and document/data processing remain local. LangSmith is used for observability and evaluation because credentials are available. When LangSmith is enabled, trace metadata/prompts/outputs may be sent to the LangSmith service; strict air-gapped/offline execution would require disabling it.


## Application

- Python 3.12+
- uv
- Streamlit

## Agent orchestration

- LangGraph
- LangChain components only where they reduce boilerplate

## Local LLM

- Ollama
- **Gemma 4**
- Prefer a lightweight Gemma 4 variant for local execution; exact variant remains configurable
- Keep the provider/model behind a small abstraction so it can be replaced without changing the LangGraph agent design

The application should keep the model provider behind a small abstraction so the model can be replaced without changing the agent graph.

## Embeddings

Fixed starting choice:

- **BGE-small**
- Run fully locally
- Keep the model configurable through `.env` so it can be benchmarked/swapped later

The project intentionally avoids large embedding models because the goal is a lightweight local interview showcase.

## Vector database

Recommended:

- **Qdrant** local mode / local service

Reason: it provides a clean path for vector search + metadata filtering without adding unnecessary application complexity.

## Keyword retrieval

- BM25
- Keep implementation simple and explicit

## Reranking

- Local cross-encoder reranker
- Use a configurable reranker model

Do not introduce advanced retrieval techniques such as HyDE, query decomposition, MMR, multi-query retrieval, etc. in V1 unless evaluation demonstrates a concrete benefit.

## Parsing

Recommended:

- **Docling** for layout-aware document parsing where practical
- PyMuPDF as a reliable PDF fallback
- python-docx / python-pptx where targeted extraction is simpler
- pandas for CSV/XLSX

## OCR / Vision

- Tesseract for OCR
- Gemma 4 / selected Ollama vision-capable model for image/chart/screenshot interpretation

OCR and visual understanding are separate steps:

```text
Image
  ├── OCR → text evidence
  └── Vision LLM → visual semantic evidence
```

## Spreadsheet reasoning

- pandas
- deterministic Python operations for calculations
- LLM generates/chooses the operation
- execution results become evidence

Avoid DuckDB in this version.

## Code analysis

- Python `ast` for Python code where useful
- deterministic extraction of functions/classes/imports
- LLM for explanation/documentation

## Observability

Use **LangSmith as a first-class observability layer**. The user already has credentials, so tracing should be enabled during development and interview demonstrations rather than treated as a later optional feature.

Recommended setup:

- LangSmith tracing enabled through environment variables
- Trace LangGraph runs and individual agent/tool calls
- Use project separation for development vs evaluation runs
- Keep sensitive uploaded document content in mind when deciding what to log
- Add an application setting to disable tracing when a strict local-only run is required

Track:


- router decision
- selected agents
- retrieval latency
- retrieved sources
- reranker outputs
- tool execution
- synthesis
- verification
- final answer

## Testing

- pytest
- unit tests
- small integration test suite
- evaluation dataset

## Optional deployment

- Run directly using uv/Python during development and demos
- Dockerize only after the local application is stable or when needed for reproducibility

## Local vs Observability Boundary

```text
Local machine
  ├── Streamlit
  ├── LangGraph
  ├── Ollama / Gemma 4
  ├── BGE-small
  ├── Qdrant
  ├── BM25
  ├── reranker
  ├── Tesseract
  ├── Docling / PyMuPDF
  └── Pandas
             │
             └── traces / run metadata (when enabled)
                              ▼
                         LangSmith
```

This distinction keeps the **AI/data workload local** while making the project easy to observe, debug, and evaluate with LangSmith.

---

# 6. Agent Design

## 6.1 Router / Planner Agent

Responsibility:

- classify user intent
- identify required evidence types
- select one or multiple agents
- decide whether a query can be answered directly

Use a hybrid approach:

```text
Deterministic rules
       +
LLM classification
       ↓
final route
```

Example:

```text
"What does the contract say about termination?"
→ Retrieval Agent

"Which region had the highest revenue?"
→ Data Agent

"What does this architecture screenshot show?"
→ Vision Agent

"Compare revenue from the PDF and Excel file."
→ Retrieval Agent + Data Agent + Synthesis
```

The router should support **parallel execution** when multiple agents are required.

---

## 6.2 Retrieval Agent

Responsibilities:

1. Query preprocessing
2. Metadata filtering
3. Semantic retrieval
4. BM25 retrieval
5. Result merging
6. Reranking
7. Evidence selection

Core V1 pipeline:

```text
Query
 ↓
Dense retrieval ───┐
                   ├─ Merge → Reranker → Evidence
BM25 retrieval ────┘
```

Avoid over-engineering this layer before evaluation.

---

## 6.3 Spreadsheet / Data Agent

Responsibilities:

- inspect workbook structure
- select relevant sheet(s)
- understand columns/types
- generate pandas operations
- execute calculations deterministically
- return structured evidence

Example:

```text
User:
"What was the highest-revenue region in Q4?"

LLM:
select relevant DataFrame + operation

Python:
df.groupby("region")["revenue"].sum().idxmax()

Agent:
return result + supporting file/sheet/cell range
```

Important principle:

> The LLM should not be trusted to perform arithmetic that Python can perform deterministically.

---

## 6.4 Vision / OCR Agent

Pipeline:

```text
Image / scanned page
        ↓
     Tesseract
        ↓
    OCR evidence
        +
 Ollama vision model
        ↓
 Visual evidence
        ↓
 Structured interpretation
```

Should handle:

- screenshots
- scanned documents
- charts
- diagrams
- forms
- visual labels
- mixed text + image pages

---

## 6.5 Code Agent

Initial scope:

- explain code
- generate documentation

Potential process:

```text
Code file
   ↓
AST / structural extraction
   ↓
functions / classes / imports
   ↓
code-aware chunks
   ↓
retrieval
   ↓
LLM explanation/documentation
```

Future capability:

- bug analysis
- refactoring
- test generation

These are not mandatory for V1.

---

## 6.6 Synthesis Agent

Responsibilities:

- combine outputs from selected agents
- resolve overlapping evidence
- preserve source attribution
- answer the actual user question
- avoid unsupported claims

The synthesis prompt must explicitly require:

```text
Use only provided evidence.
Do not invent missing facts.
Keep citations attached to claims.
State when evidence is insufficient.
```

---

## 6.7 Verification / Citation Agent

Responsibilities:

- check whether answer claims are supported
- verify citations
- detect unsupported statements
- request regeneration or downgrade confidence when evidence is weak

Potential flow:

```text
Draft answer
     ↓
Extract claims
     ↓
Find supporting evidence
     ↓
Check support
     ↓
Pass / regenerate / abstain
```

The system should prefer:

> "I couldn't find sufficient evidence in the uploaded files."

over fabricated answers.

---

# 7. Canonical Evidence Model

Every extracted item should carry source metadata.

Conceptual schema:

```python
Evidence(
    document_id=...,        # stable document/session identifier
    file_name=...,          # original filename
    file_type=...,          # pdf/docx/xlsx/image/code/etc.
    source_type=...,        # paragraph/table/image/code/dataframe
    page=...,               # when applicable
    section=...,            # when available
    sheet=...,              # spreadsheet only
    cell_range=...,         # spreadsheet only
    chunk_id=...,           # retrieval identifier
    content=...,            # extracted content
    metadata={...}
)
```

This should be the foundation for precise citations.

---

# 8. Citation Strategy

Target citation precision:

### PDF/DOCX/PPTX

```text
[file.pdf — Page 12 — Section: Revenue Outlook]
```

### Spreadsheet

```text
[sales.xlsx — Sheet: Q4 — Range: A12:F28]
```

### Image

```text
[architecture.png — OCR + visual region]
```

### Code

```text
[app.py — function: `build_rag_graph()`]
```

Citation generation should happen from metadata attached to evidence, never from manually generated strings after the fact.

---

# 9. Conversation Memory

Memory is **session-only**.

Recommended state model:

```text
SessionState
 ├── uploaded_documents
 ├── indexes
 ├── document_metadata
 ├── conversation_messages
 ├── active_context
 ├── agent_results
 └── citation/evidence state
```

Do not persist conversations between application restarts in V1.

Do not blindly send the entire conversation to the LLM on every turn.

Use a bounded working context and summarize older turns when necessary.

---

# 10. Handling 50 Files

Initial constraint:

- up to 50 files per session
- each file < 50 MB

Important engineering concerns:

### Do not process everything synchronously on every query

Instead:

```text
Upload
 ↓
Ingest once
 ↓
Create indexes / structured representations
 ↓
Reuse them for all chat turns
```

### Metadata isolation

Every chunk/vector should include a session/document identifier so retrieval can be filtered correctly.

### Large-file handling

For large documents:

- process incrementally
- avoid loading every representation into the prompt
- retrieve only relevant evidence

---

# 11. Graceful Failure / Abstention

The system should explicitly support:

```text
No relevant evidence found
        ↓
Do not answer from model memory
        ↓
Return grounded abstention
```

Example:

> I couldn't find sufficient evidence in the uploaded files to answer this reliably.

For mixed-evidence cases:

> The uploaded files support X, but I could not verify Y.

This is a key interview talking point around hallucination control.

---

# 12. Streamlit Showcase UI

Recommended layout:

```text
┌──────────────────┬──────────────────────────────────────┐
│ Files            │ Chat                                 │
│                  │                                      │
│ report.pdf       │ User question                        │
│ sales.xlsx       │                                      │
│ diagram.png      │ Assistant answer                    │
│ code.py          │                                      │
│                  │ Sources / citations                  │
│ [+ Upload]       │                                      │
│                  │ ──────────────────────────────────── │
│                  │ Evidence / Agent trace (expandable)  │
└──────────────────┴──────────────────────────────────────┘
```

Useful showcase features:

- multi-file upload
- file list with type/size/status
- chat history
- streaming answer
- inline citations
- expandable source evidence
- optional agent routing trace
- optional confidence/evidence indicator
- table rendering for structured answers

For an interview demo, **agent trace and citations should be visible but not forced into the main conversational answer**.

---

# 13. Suggested Repository Structure

```text
multi-agent-document-chat/
│
├── pyproject.toml
├── uv.lock
├── README.md
├── .env.example
├── .gitignore
│
├── app/
│   ├── main.py
│   │
│   ├── agents/
│   │   ├── router.py
│   │   ├── retrieval.py
│   │   ├── data_agent.py
│   │   ├── vision_agent.py
│   │   ├── code_agent.py
│   │   ├── synthesis.py
│   │   └── verification.py
│   │
│   ├── graph/
│   │   ├── state.py
│   │   ├── graph.py
│   │   └── routing.py
│   │
│   ├── ingestion/
│   │   ├── router.py
│   │   ├── pdf_parser.py
│   │   ├── document_parser.py
│   │   ├── spreadsheet_parser.py
│   │   ├── image_parser.py
│   │   ├── code_parser.py
│   │   └── canonical.py
│   │
│   ├── retrieval/
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── bm25.py
│   │   ├── reranker.py
│   │   └── hybrid.py
│   │
│   ├── tools/
│   │   ├── pandas_tools.py
│   │   ├── ocr_tools.py
│   │   ├── code_tools.py
│   │   └── document_tools.py
│   │
│   ├── memory/
│   │   └── session.py
│   │
│   ├── models/
│   │   ├── llm.py
│   │   ├── embeddings.py
│   │   └── schemas.py
│   │
│   ├── evaluation/
│   │   ├── datasets/
│   │   ├── metrics.py
│   │   └── runner.py
│   │
│   └── utils/
│       ├── logging.py
│       └── config.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── evals/
│   └── questions.json
│
├── sample_data/
│
└── docker/
    └── Dockerfile
```

The structure is intentionally modular so every major interview topic has a clear location in the codebase.

---

# 14. Development Roadmap

## Milestone 1 — Skeleton

- initialize project with uv
- configure Streamlit
- configure Ollama
- configure LangSmith
- define LangGraph state
- create basic chat flow

## Milestone 2 — Document ingestion

Implement:

- PDF
- DOCX
- TXT
- Markdown
- PPTX

Create canonical evidence objects and metadata.

## Milestone 3 — Retrieval

Implement:

- embeddings
- Qdrant
- BM25
- hybrid retrieval
- reranking
- metadata filtering

## Milestone 4 — Grounded document chat

Implement:

- Retrieval Agent
- Synthesis Agent
- citations
- abstention

At this point we already have a strong baseline application.

## Milestone 5 — Multi-agent routing

Implement:

- hybrid router
- LangGraph conditional routing
- parallel agents
- cross-document synthesis

## Milestone 6 — Spreadsheet agent

Implement:

- CSV/XLSX ingestion
- DataFrame registry
- Pandas reasoning
- deterministic calculations
- sheet/range citations

## Milestone 7 — Vision/OCR

Implement:

- Tesseract
- visual model integration through Ollama
- image evidence objects
- chart/screenshot reasoning

## Milestone 8 — Code agent

Implement:

- Python AST parsing
- code-aware chunks
- explanation
- documentation generation

## Milestone 9 — Verification

Implement:

- claim extraction
- evidence verification
- citation validation
- abstention/regeneration path

## Milestone 10 — Evaluation

Create a controlled dataset and measure:

- retrieval Recall@K
- MRR / ranking quality
- answer correctness
- faithfulness / groundedness
- citation precision
- citation recall
- router accuracy
- latency
- failure/abstention behavior

## Milestone 11 — Showcase polish

- Streamlit UX
- streaming
- source viewer
- agent trace
- error handling
- README architecture diagram
- demo dataset
- benchmark results
- optional Docker support

---

# 15. What Makes This Interview-Worthy

The project should be presented as an engineering system, not as a chatbot UI.

Key discussion points:

### Multi-agent architecture

Why different tasks require different reasoning strategies.

### Hybrid retrieval

Why semantic search alone is insufficient for names, identifiers, exact phrases, and structured evidence.

### Reranking

Why candidate retrieval and final evidence selection are separate stages.

### Structured data reasoning

Why spreadsheets should be handled by deterministic Python rather than treated as plain text.

### Vision + OCR

Why OCR text extraction and visual interpretation solve different problems.

### Grounding

Why citations must originate from evidence metadata.

### Verification

Why generated answers should be checked against retrieved evidence.

### Abstention

Why a trustworthy system should know when not to answer.

### Performance

Why ingestion is done once and reused across conversation turns.

### Observability

Why LangSmith is useful for debugging agent routing, retrieval and generation.

### Local privacy

Why Ollama + local embeddings + local retrieval keep the main AI/data workload on-device. When LangSmith tracing is enabled, selected trace data can leave the machine for observability; strict air-gapped/offline mode requires tracing to be disabled.

---

# 16. Explicitly Out of Scope for V1

To keep the project focused:

- Authentication
- Persistent user accounts
- Persistent chat history
- DuckDB
- Cloud-hosted LLM APIs
- Multiple complex retrieval strategies such as HyDE/multi-query/MMR by default
- Full web search
- Audio/video ingestion
- Full code execution sandbox
- Autonomous code modification
- Enterprise permissions / RBAC

These can be future extensions but should not distract from the interview showcase.

---

# 17. Open Decisions Still Needed

Only a few choices remain before implementation should start.

### 1. Vector DB execution mode

Recommended:

- Qdrant local service via Docker, while the rest of the app runs directly with uv.

Alternative:

- Qdrant local embedded mode if supported cleanly by the chosen client/version.

### 2. Embedding model

**Locked:** BGE-small. The implementation should still keep the embedding interface swappable for later benchmarking.

### 3. Reranker model

Pick one small local cross-encoder suitable for the RTX 5050.

### 4. Vision model

Use the selected Gemma 4 / Ollama vision-capable setup where supported; keep the vision adapter configurable so it can be swapped independently later.

### 5. Initial code-language scope

Recommended V1:

- Python only

because AST-based processing gives a stronger demonstration of code-aware ingestion without spreading implementation effort across many languages.

### 6. Evaluation dataset

We should decide whether to create:

- a fully synthetic demo corpus,
- a small manually curated corpus,
- or a mixed corpus.

Recommended for interviews: **mixed corpus with a manually curated evaluation set**.

---

# 18. Recommended First Implementation Target

Start with this vertical slice before adding all agents:

```text
Streamlit
   ↓
Upload PDF/DOCX/PPTX/TXT
   ↓
Layout-aware ingestion
   ↓
Canonical evidence
   ↓
Embeddings + BM25
   ↓
Hybrid retrieval
   ↓
Reranking
   ↓
Retrieval Agent
   ↓
Synthesis Agent
   ↓
Verification Agent
   ↓
Precise citations
```

Once this is reliable, add:

```text
Data Agent
Vision Agent
Code Agent
```

This ordering minimizes integration risk while still producing a meaningful demo early.

---

# 19. Interview Demo Script

A strong demo should use one session containing several related files:

```text
annual_report.pdf
financials.xlsx
architecture.png
pipeline.py
company_notes.docx
presentation.pptx
```

Then demonstrate progressively:

1. Ask a normal document question.
2. Ask a cross-document question.
3. Ask an exact spreadsheet calculation.
4. Ask the system to interpret an architecture/image.
5. Ask it to explain a Python file.
6. Ask a question with insufficient evidence and show abstention.
7. Show exact citations.
8. Open LangSmith and show the router → agents → retrieval → verification trace.

That tells a much stronger interview story than simply showing a chat interface.

---

# 20. Success Criteria

The project is considered successful when it can reliably demonstrate:

- multi-format ingestion
- 50-file session design
- local LLM inference with Ollama + Gemma 4
- hybrid retrieval + reranking
- multi-agent routing
- cross-document reasoning
- real spreadsheet calculations
- OCR + visual understanding
- code explanation/documentation
- precise citations
- grounded verification
- graceful abstention
- session-only conversational context
- LangSmith traces
- reproducible local setup with uv
- measurable evaluation results

The emphasis should be on **correctness, explainability, and engineering depth**, not maximum number of features.
