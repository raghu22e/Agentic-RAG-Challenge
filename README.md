<img width="1883" height="744" alt="image" src="https://github.com/user-attachments/assets/08335ac0-570a-4250-9ba1-c7c14b1639a9" /># Agentic RAG — Prerequisite & Course Planning Assistant

End-to-end **RAG + multi-agent** assistant that ingests **your PDFs** under `data/` (**academic policy**, **courses**, **programme requirements / B.Tech eligibility**), answers **prerequisite** and **program** questions with **citations**, suggests **course plans**, and **abstains** when evidence is missing. **Flask** UI shows **retrieved chunks**, **source paths**, **highlight-style** excerpt panels, **structured sections**, and **agent reasoning steps**.

## Architecture

| Stage | Agent | Role |
|--------|--------|------|
| 1 | **Intake** | Updates student profile JSON; proposes clarifying questions |
| 2 | **Catalog Retriever** | Embedding search (`sentence-transformers`) over **FAISS** |
| 3 | **Planner** | LLM with strict prompt: citations, structured sections, no guessing |
| 4 | **Verifier** | Rule check (course codes must appear in retrieved chunks); optional LLM revision |

**Stack:** Python · `pypdf` ingestion · `sentence-transformers` · `faiss-cpu` · `langchain-core` + **`langchain-openai`** ( **[Sarvam AI](https://docs.sarvam.ai/api-reference-docs/getting-started/quickstart)** by default, or OpenAI) · Flask.

> **Note:** The repo avoids `langchain-community` PDF loaders so ingestion does not pull the heavy NLTK/html toolchain (fewer install failures on constrained machines).

## Directory layout

```text
project/
  main.py                 # CLI: generate-data | ingest | evaluate | serve
  requirements.txt
  README.md
  .env.example            # copy to .env — set SARVAM_API_KEY or OPENAI_API_KEY
  data/
    README.md               # where to put PDFs
    academic_policy/        # regulations, attendance, exams, etc.
    courses/                # per-course PDFs (e.g. 12 subjects)
    programme_requirements/ # B.Tech eligibility, degree rules
    generate_synthetic_catalogs.py   # optional demo PDFs → data/pdfs/
    pdfs/                   # optional legacy flat folder
  src/
    config.py
    documents.py
    ingestion.py
    llm_provider.py
  embeddings/
    embedder.py
  vectorstore/
    faiss_store.py
  retriever/
    catalog_retriever.py
  prompts/
    templates.py
  agents/
    intake_agent.py
    planner_agent.py
    verifier_agent.py
    parse_response.py
    orchestrator.py
  evaluation/
    test_dataset.json
    evaluate.py
  frontend/
    app.py
    templates/index.html
    static/style.css
```

## Quick start

### 1. Environment (recommended)

Use a **virtual environment** with enough free disk (~2–4 GB) for PyTorch/sentence-transformers wheels.

```bash
cd project
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Add your PDFs and build the index

Put files here (see `data/README.md`). Names like `Academic Policy` / `Programme Requirements` work; types are inferred from the folder name.

```text
data/academic_policy/          ← policy PDF(s)
data/courses/                  ← course PDFs (e.g. 12 courses)
data/programme_requirements/   ← B.Tech eligibility / programme PDF(s)
```

Then:

```bash
python main.py ingest          # indexes all PDFs above (+ optional data/pdfs/)
```

Optional synthetic demo PDFs:

```bash
python main.py generate-data   # writes under data/pdfs/ only
```

### 3. Configure the chat LLM (Sarvam or OpenAI)

**Default path — [Sarvam AI](https://dashboard.sarvam.ai/key-management)** (Indian provider, OpenAI-compatible HTTP API per [their quickstart](https://docs.sarvam.ai/api-reference-docs/getting-started/quickstart)):

1. Copy `.env.example` to **`.env`** in the project root.
2. Set **`SARVAM_API_KEY`** (from [Sarvam key management](https://dashboard.sarvam.ai/key-management)).  
   **Do not** put a Sarvam key in **`OPENAI_API_KEY`** — OpenAI’s servers will return **401 invalid_api_key**.

Optional: **`SARVAM_CHAT_MODEL`** (default `sarvam-30b`), **`SARVAM_API_BASE`** (default `https://api.sarvam.ai/v1`).

**Alternative — OpenAI** (only if `SARVAM_API_KEY` is unset):

```bash
set OPENAI_API_KEY=sk-...
```

Default OpenAI model: **`gpt-4o-mini`** (`OPENAI_CHAT_MODEL`).

**`python-dotenv`** loads `.env` when you run `main.py` or `frontend/app.py`.

**Demo / no API (UI + retrieval only)**

```bash
set DEMO_MODE=1
```

Returns a **placeholder** answer; **retrieval and verifier rules** still run.

### 4. Run the Flask UI

```bash
python main.py serve
```

Open `http://127.0.0.1:5000`. Use **Chat**, optional **Upload PDF** (rebuilds index), or **Reindex all PDFs**.

#### Speed

The **first** chat after start loads the embedding model once (~tens of seconds). Later requests reuse it.

For **faster answers** (fewer LLM round-trips, smaller retrieval window):

```powershell
# PowerShell
$env:RAG_FAST="1"
$env:RAG_SKIP_INTAKE_LLM="1"
$env:LLM_TIMEOUT_SEC="90"
python main.py serve
```

- **`RAG_FAST=1`** — skips the verifier’s **second** LLM call (rule-based verify only); uses fewer chunks (`TOP_K` default 5).
- **`RAG_SKIP_INTAKE_LLM=1`** — skips the intake LLM (profile is not inferred from chat; retrieval still uses your message).

#### Missing API key / connection issues

| Situation | What happens |
|-----------|----------------|
| No `SARVAM_API_KEY` / `OPENAI_API_KEY` and no `DEMO_MODE` | **Demo LLM** with a hint to set keys. Retrieval still works. |
| Invalid or blocked OpenAI access | You may see HTTP errors in the Flask terminal; fix the key or network. |

| Situation | Fix |
|-----------|-----|
| **`WinError 10061`** on chat | Usually **not** OpenAI (HTTPS). If you see it, check nothing else is misconfigured; ensure Flask is running. |
| Browser cannot load the app | Run `python main.py serve` and open `http://127.0.0.1:5000`. |

**`GET /api/health`** reports **`backend`**: `openai` | `demo` (and `reason` when demo).

#### UI error: `Unexpected token '<'` / not valid JSON

That means the browser got **HTML** (e.g. Flask error page or wrong server), not JSON. Fixes:

- Ensure the URL is the Flask app (`http://127.0.0.1:5000`), not another site.
- Check the **terminal** running Flask for tracebacks.
- The UI now detects HTML responses and shows a clearer message; chat requests **time out** after 3 minutes by default (edit `CHAT_TIMEOUT_MS` in `frontend/templates/index.html` if needed).

### 5. Evaluation script

```bash
python main.py evaluate
# or: python evaluation/evaluate.py
```

Reports:

- **Citation coverage rate** — citations mention a retrieved source filename (heuristic)
- **Accuracy** — gold **substring** checks on non–`not_in_docs` items
- **Abstention accuracy** — `not_in_docs` items should contain abstention-style phrasing

Use a **real LLM** for meaningful accuracy; with `DEMO_MODE=1`, substring accuracy will be low.

## Response format (enforced in prompts)

```text
Answer / Plan:
Why:
Citations:
Clarifying Questions:
Assumptions:
```

## Dataset (synthetic)

`data/generate_synthetic_catalogs.py` creates:

- **22** course-catalog-style PDFs (`vit_sample_catalog_*.pdf`)
- **2** program requirement PDFs (`anna_program_btech_cse_requirements.pdf`, `iit_sample_program_ece_requirements.pdf`)
- **1** academic policy PDF (`nist_sample_academic_policy.pdf`)

Content is **fictional** but structured like Indian UG handbooks for the assignment.

## Frontend visualization

- **Retrieved chunks** with **source filename**, **page**, **chunk id**, **similarity score**
- **Highlighted** excerpt cards (accent border / background)
- Parsed **Answer / Plan**, **Citations**, etc.
- **Ordered reasoning steps** (intake → retrieve → plan → verify)

## Troubleshooting

| Issue | Suggestion |
|--------|------------|
| `No space left on device` during `pip install` | Free disk or use a smaller env; install on another drive |
| `FAISS index missing` | Run `python main.py ingest` |
| NumPy / pandas binary errors in **global** Python | Always use the project **venv** |
| OpenAI errors | Verify `OPENAI_API_KEY`, billing, and network; check Flask logs |

##Example Input
Q.)What programs do u offer??
Ans.)Based on the catalog excerpts, the program you're interested in is the B.Tech in Computer Science Engineering.

Here are the key details about the program:

*   **Program:** B.Tech Computer Science Engineering
*   **Total Credits Required:** 160 credits
*   **Program Structure:** The program is structured into Core Courses, AI/ML Electives, and Open Electives.
*   **AI/ML Electives:** You can choose any two from a list that includes Machine Learning (CSE301), Artificial Intelligence (CSE302), Deep Learning (CSE303), and Data Mining (CSE304).
*   **Project Requirement:** You must complete a Final Year Project worth 6 credits.
*   **Internship Requirement:** A mandatory 8-week internship is required.
  

