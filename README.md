# BD Demo Project

A local-first document intelligence demo with Streamlit, retrieval, spreadsheet reasoning, and multi-agent routing.

## Prerequisites

Before running the app, make sure you have:

- Python 3.13+
- `uv` installed
- Ollama installed and running locally
- Optional: Tesseract OCR installed for image/OCR parsing

## 1) Open the project folder

```powershell
cd c:\Users\innov\Desktop\Project\BD_demo_prjt
```

## 2) Create and activate a virtual environment

Using `uv`:

```powershell
uv venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## 3) Install dependencies

```powershell
uv pip install -e .
```

This will install the dependencies declared in `pyproject.toml`.

## 4) Configure environment variables

Create a local `.env` file in the project root if you want to override defaults. Example:

```env
LLM_MODEL=deepseek-r1:1.5b
OLLAMA_BASE_URL=http://localhost:11434
ENABLE_LANGSMITH=false
LANGCHAIN_TRACING=false
LANGCHAIN_PROJECT=demo-prjt-observation
```

The project already has defaults in [app/config.py](app/config.py), so this file is optional for basic usage.

## 5) Start Ollama and pull a model

If you want full LLM-based chat support:

```powershell
ollama serve
ollama pull gemma4:e4b
```

You can change the model in `.env` or in the config settings.

## 6) Start the app

From the project root:

```powershell
streamlit run app/main.py
```

Then open the local URL shown by Streamlit in the browser.

## 7) Use the app

- Upload files such as `.txt`, `.md`, `.csv`, `.xlsx`, `.png`, `.jpg`, `.py`
- Ask questions in the chat
- The app routes document, spreadsheet, and multi-agent questions automatically
- Evidence and trace details are surfaced in the demo UI

## 8) Optional: OCR support

If you want image OCR to function locally, install Tesseract:

- Windows: install the official Tesseract build and ensure it is on PATH
- Then restart the terminal before running the app

## 9) Optional: run tests

```powershell
python -m pytest -q
```

## 10) Common troubleshooting

### Streamlit command not found
```powershell
uv pip install streamlit
```

### Ollama model not found
```powershell
ollama pull gemma4:e4b
```

### OCR not working
- Ensure Tesseract is installed
- Confirm `tesseract --version` works in a new terminal

### Local model not responding
- Check that Ollama is running with `ollama serve`
- Verify the model name matches your local installation

## Project overview

This demo includes:

- document ingestion and retrieval
- hybrid retrieval logic
- spreadsheet/data reasoning
- multi-agent route selection
- verification and grounded answer flow
- local-first demo UI

## Notes

This is designed as a local-first interview/demo project. The app can run without cloud services, but Ollama is required for the full LLM-backed experience.
