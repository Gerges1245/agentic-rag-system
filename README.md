# Agentic RAG System

An educational document assistant that plans how to answer a question, chooses tools, and returns answers with supporting sources. The application uses a Gradio interface, Gemini, Sentence Transformers, and a persistent ChromaDB collection.

## Features

- Upload and index PDF, DOCX, TXT, and CSV files for a selected course.
- Extract and clean text, split it into overlapping chunks, generate embeddings, and store them in ChromaDB.
- Search indexed course materials and refine a search through additional tool calls.
- Plan and execute tool calls with a configurable limit of up to six steps in the interface.
- Calculate arithmetic expressions, inspect uploaded CSV files, and compute filtered aggregates.
- Summarize the indexed text of a specific PDF, DOCX, or TXT document with source references.
- Display the answer, public action plan, executed actions, and cited source quotations.
- Return an insufficient-evidence response when the model determines that the evidence does not support the question.

The application validates citation identifiers and checks that supporting quotations occur in their sources. These checks do not prove that every claim is entailed by its citation; review important answers against the sources.

## How it works

1. The user selects a course and asks a question.
2. The agent prepares a short plan and selects tools based on the question.
3. The application executes the chosen tools, returns their results to the model, and records the actions.
4. The model prepares an evidence-based answer. The application checks the citations and quotations before showing it.

Available tools include `search_materials`, `calculate`, `list_csv_files`, `analyze_csv`, and `summarize_document`. CSV analysis uses the saved CSV file for the selected course. For summarization, the source filename must match an indexed PDF, DOCX, or TXT document. Open **Browse available sources** and select **Show available sources** to find the saved filename.

The interface allows up to six tool attempts per question. The summarization tool may make additional Gemini requests internally when a document spans multiple batches; it supports up to four batches and fails explicitly if its configured input limit is exceeded. Summary coverage refers to indexed text. Image-only content that was not extracted is outside that coverage.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Gradio interface, document upload, answers, plans, actions, and sources. |
| `agentic_rag.py` | Planning, tool execution loop, evidence handling, and answer validation. |
| `agent_tools.py` | Course-material search and arithmetic calculation. |
| `csv_tools.py` | CSV discovery, inspection, filtering, and numeric aggregation. |
| `document_tools.py` | Document discovery and indexed-text summarization. |
| `ingestion.py` | Document extraction, cleaning, chunking, embedding, and indexing. |
| `rag_core.py` | Shared course configuration, Gemini client, embedding model, and ChromaDB access. |
| `requirements.txt` / `requirements-lock.txt` | Dependencies and the recorded environment versions. |
| `data/` / `vector_db/` | Local documents and the persistent knowledge base. |
| `tests/` | Individual tool and agent integration checks. |
| `examples/` | Sample documents used during development. |

## Setup on Windows

Python 3.11 was used for the working environment. In a PowerShell terminal opened in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements-lock.txt
```

Create a local `.env` file based on `.env.example` and set `GEMINI_API_KEY`. An optional `GEMINI_MODEL` can be set to a model available to the account. The application needs internet access for Gemini; the embedding model may also need an initial download.

Keep the API key private. The repository's `.gitignore` excludes `.env`, `.venv/`, `data/`, and `vector_db/`.

## Knowledge base

The current code expects the complete persistent database directory at:

```text
vector_db/chroma.sqlite3
```

It opens the collection `course_materials_all_formats_v1` and uses the embedding model `sentence-transformers/all-MiniLM-L6-v2`. Preserve the whole `vector_db` directory, including its associated index files. The application expects this collection to exist; it does not automatically create a replacement database when it is missing.

Because `vector_db/` and `data/` are ignored by Git, a repository copy alone does not include the knowledge base or original uploaded CSV files. Obtain the complete compatible database and document files separately, or prepare a compatible collection and index the documents before running the application. Keep the same embedding model when querying existing vectors.

Available course selections are **Python Programming**, **Database Systems**, and **Computer Networks**. Uploads are saved under the selected course. CSV analysis reads uploaded CSV files in that course's upload directory.

## Run the app

```powershell
.\.venv\Scripts\python.exe app.py
```

Open the local address printed in the terminal and keep the terminal running. Select a course, upload a document if needed, and ask a question. The interface shows answers and sources alongside the question. Open **Agent plan and actions** to inspect the plan and the executed tools. The upload result and **Browse available sources** display saved filenames for use in summary questions.

Example questions:

- `What is a primary key?`
- `Calculate (120 + 80) / 4.`
- `Summarize the indexed document <exact saved filename>.`
- `Find <saved CSV filename>, calculate the mean of a numeric column, and report the result with sources.`

## Validation

The working project has been exercised through the terminal and Gradio for document search, arithmetic, CSV analysis, document summarization, a multi-step CSV-and-calculation question, document upload, and an insufficient-evidence question. A missing-evidence test also showed a second search with a revised query.

The checked CSV fixture `agentic_scores.csv` produced an overall mean of 80, a group A mean of 90, and a difference of 10. These values describe that test file, not arbitrary uploaded data.

The following scripts exercise individual components and integrations. Several use Gemini and may consume API quota; some depend on the existing database and the specific local fixture files created during development. Run these commands from the project root. The first command adds the project root to Python's import path for the current PowerShell session.

```powershell
$env:PYTHONPATH = "$((Get-Location).Path);$env:PYTHONPATH"
.\.venv\Scripts\python.exe .\tests\test_tools.py
.\.venv\Scripts\python.exe .\tests\test_agentic.py search
.\.venv\Scripts\python.exe .\tests\test_agentic.py calc
.\.venv\Scripts\python.exe .\tests\test_csv_tools.py
.\.venv\Scripts\python.exe .\tests\test_agentic_csv.py
.\.venv\Scripts\python.exe .\tests\test_document_summary.py
.\.venv\Scripts\python.exe .\tests\test_agentic_summary.py
.\.venv\Scripts\python.exe .\tests\test_agentic_multistep.py
```

Live model output can vary. A comparison of primary and foreign keys was tested through Gradio; its foreign-key explanation was brief, so generated explanations should be reviewed against their cited source material.

## Troubleshooting

- **Missing key:** check the local `.env` file and restart the app after editing it.
- **Missing database or collection:** restore the complete compatible `vector_db` directory and verify the collection name above.
- **No matching CSV:** confirm the course and use **Browse available sources**, then **Show available sources**, to copy the exact saved filename.
- **No extracted PDF text:** ensure the PDF contains extractable text; scanned-image OCR is not part of the current pipeline.
- **Gemini returns 503:** the service may be temporarily unavailable; retry after a short wait.
- **Insufficient evidence or invalid citations:** review the action trace and source material, and retry or refine the question. The application is designed to avoid presenting an unchecked answer as supported.

## Author

Gerges William