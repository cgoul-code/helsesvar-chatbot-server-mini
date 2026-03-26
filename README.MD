# HEI 2.0 — KI-chatbot backend

Backend for the AI chatbot in **«Hva er innafor» 2.0 (HEI 2.0)**, developed by
Helsedirektoratet. The chatbot answers questions about relationships, sexuality,
feelings, and sexual health aimed at Norwegian youth aged 13–19.

---

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agent workflow](#agent-workflow)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Running locally](#running-locally)
- [API endpoints](#api-endpoints)
- [Session management](#session-management)
- [Severity classification](#severity-classification)
- [Adding new content](#adding-new-content)
- [Deployment](#deployment)

---

## Overview

The backend is a streaming HTTP API built with **Quart** (async Flask). It
exposes three endpoints consumed by the HEI 2.0 frontend:

| Endpoint | Purpose |
|---|---|
| `POST /chat` | Main Q&A — returns a grounded, empathy-rewritten answer |
| `POST /examples` | Returns categorised example questions for the landing page |
| `POST /chat` (agent `hvaerinnafor_related_qa`) | Returns a pre-written answer for a question clicked from the related-questions list |

Answers are streamed as **Server-Sent Events (SSE)** so the frontend can render
text progressively.

---

## Architecture

```
Frontend
   │  SSE stream
   ▼
Quart HTTP server  (app.py / routes.py)
   │
   ├── /chat  ──────────────────────────────────────────────────┐
   │                                                            │
   │   answer_utils.py                                          │
   │       └── get_answer_as_stream()                          │
   │               │                                            │
   │               ▼                                            │
   │         LangGraph workflow  (agent_workflow_answer.py)    │
   │               │                                            │
   │         ┌─────┴──────┐                                    │
   │    analyze_query   (rewrite + severity)                    │
   │         │                                                  │
   │    ┌────┴─────┐                                            │
   │ fast_single  orchestrator  (multi-subquery)                │
   │    │              │                                        │
   │    │         query_grounded  ×N  (parallel workers)        │
   │    │              │                                        │
   │    │         synthesizer                                   │
   │    └──────────────┘                                        │
   │         │                                                  │
   │    empathy_rewrite                                         │
   │         │                                                  │
   │    emit_query_answer_references                            │
   │         │                                                  │
   │    related_queries_dialog_from_query                       │
   │                                                            │
   ├── /chat  (agent: hvaerinnafor_related_qa) ────────────────┘
   │   agent_workflow_qa.py  →  fetch pre-written answer by node_id
   │
   └── /examples
       answer_utils.py  →  get_examples_full_as_stream()
```

**Vector indexes** (LlamaIndex, persisted to Azure Blob Storage mount):

| Index name | Content |
|---|---|
| `hvaerinnafor` | Source articles — the factual knowledge base |
| `hvaerinnafor_qa_bank` | Pre-written Q&A pairs used for related questions and fast answers |

---

## Agent workflow

The main `/chat` workflow (`agent_workflow_answer.py`) runs as a LangGraph state
machine:

```
START
  └─► analyze_query          Rewrites query, detects severity (Green/Yellow/Red),
                              decides if subqueries are needed
        ├─► fast_single       Single question — one worker, direct answer
        └─► orchestrator      Multi-topic — splits into subqueries
              └─► query_grounded  ×N   Retrieval → grounded answer → citation check
                    └─► synthesizer   Merges sub-answers into one coherent response
                          └─► empathy_rewrite   Adds warmth/empathy based on severity
                                └─► emit_query_answer_references   Streams answer + refs
                                      └─► related_queries_dialog_from_query   Suggests follow-up Qs
END
```

### Grounding and citation verification

Each worker node (`query_grounded`) retrieves source nodes, generates a
`GroundedAnswer` (answer + claims + citations), then verifies every citation
quote against the retrieved text. A subquery answer is accepted if at least 50%
of its claims have at least one verified citation.

### Severity levels

| Level | Meaning | Example |
|---|---|---|
| **Green** | General information, preventive | "What is consent?" |
| **Yellow** | Difficult situation, needs empathy | "I regret sending a nude photo" |
| **Red** | Acute crisis, needs immediate help | "I was assaulted yesterday" |

Severity affects which related questions are shown, the empathy rewrite
intensity, and which fallback placeholder is used when no answer is found.

---

## Project structure

```
.
├── app.py                        Quart app entry point
├── routes.py                     HTTP route handlers + session management
├── config.py                     LLM + embedding setup, index loading
├── query_utils.py                QuerySettings dataclass
├── answer_utils.py               Agent entry points (stream wrappers)
├── agent_workflow_answer.py      Main LangGraph Q&A workflow
├── agent_workflow_qa.py          Related-question lookup workflow
├── agent_shared.py               Shared helpers (emit, normalize, retriever builder)
├── registry.py                   All prompt templates (GROUNDED_PROMPT, EMPATHY_REWRITE_PROMPT, …)
├── json_utils.py                 Safe JSON parsing for LLM outputs
├── graph_utils.py                Mermaid diagram export helper
├── requirements.txt
└── .env                          Environment variables (not committed)
```

---

## Prerequisites

- Python 3.11+
- Access to **Azure OpenAI** (chat completion + embeddings deployments)
- Vector indexes built and available at the paths in `VECTOR_INDEX_MAP` (see
  `config.py`), either locally or mounted from Azure Blob Storage

---

## Configuration

Copy `.env.example` to `.env` and fill in all values:

```env
# Azure OpenAI — chat completion
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/

# Azure OpenAI — embeddings
AZURE_OPENAI_EMBEDDINGS_MODEL=text-embedding-3-large
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_EMBEDDINGS_API_KEY=<your-key>
AZURE_OPENAI_EMBEDDINGS_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_EMBEDDINGS_API_VERSION=2024-02-01

# Cost tracking (USD per 1 million tokens)
PRICE_INPUT_USD_PER_M=0.15
PRICE_OUTPUT_USD_PER_M=0.60
```

---

## Running locally

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# edit .env with your Azure credentials

# 4. Place vector indexes at the expected paths (or symlink them):
#    ./blobstorage/chatbot/hvaerinnafor
#    ./blobstorage/chatbot/hvaerinnafor_qa_bank

# 5. Start the server
python app.py
# Server starts at http://0.0.0.0:80
```

The server loads indexes asynchronously after startup. The `/chat` endpoint
returns HTTP 503 until indexes are ready.

---

## API endpoints

### `POST /chat`

Streams an answer to a user question.

**Request body:**
```json
{
  "agent": "hvaerinnafor",
  "vectorIndex": "hvaerinnafor",
  "query": "Hva er egentlig forelskelse?",
  "session_id": "abc123",
  "messages": [
    { "role": "user", "content": "Hva er egentlig forelskelse?" }
  ],
  "similarity_top_k": 10,
  "similarity_cutoff": 0.45
}
```

**SSE events emitted:**

| Event name | Content |
|---|---|
| `answer` | Answer text delta (markdown) |
| `short_answer` | One-sentence summary |
| `references` | `[Title](url) \|\|IMG\|\| icon_url` per source |
| `related queries` | JSON array of follow-up question suggestions |
| `Refined query` | Rewritten version of the user's question |
| `info` | Internal pipeline step names (for debugging) |
| `systeminfo` | Citation verification details (for debugging) |
| `Token usage` | Token counts and estimated NOK cost |
| `done` | Stream completed |
| `error` | Error message if something failed |

To use the **related-question fast path**, set `agent` to
`hvaerinnafor_related_qa` and include `from_node_id` (the node ID from a
previously emitted `related queries` item).

### `POST /examples`

Returns categorised example questions for the landing page. No streaming — emits
a single `examples_categories` event followed by `done`.

**Request body:**
```json
{
  "agent": "hvaerinnafor_examples",
  "vectorIndex": "hvaerinnafor",
  "requested_categories": ["Forelskelse / Flørting", "Sex / Intimitet"]
}
```

**Response event:**
```json
{
  "event": "examples_categories",
  "items": [
    {
      "id": "forelskelse-/-florting",
      "title": "Forelskelse / Flørting",
      "items": [
        { "query": "Hvordan vet jeg om jeg er forelsket?", "node_id": "abc" }
      ]
    }
  ]
}
```

---

## Session management

Conversation history is stored server-side in a **diskcache** store
(`./session_cache`), keyed by `session_id`. The frontend receives the
`session_id` in the `open` event and should include it in subsequent requests.

History is automatically trimmed to stay within:
- **20 messages** maximum
- **8 000 characters** total

Trimming always preserves complete user/assistant pairs and ensures history
starts with a user message.

---

## Severity classification

The `analyze_query` node classifies every incoming question into one of three
severity levels before retrieval. This affects:

- Which related questions are surfaced (Green questions only see Green
  candidates; Yellow sees Green + Yellow; Red sees all)
- The tone of the empathy rewrite step
- Which fallback placeholder is shown when no answer is found

For **Red** severity questions, the fallback always directs the user to
professional help resources (e.g. Sex og samfunn).

---

## Adding new content

1. **New articles** — add source documents to the `hvaerinnafor` index using
   the existing ingestion pipeline. Each document needs `title`, `url`, and
   optionally `icon_url` in its metadata.

2. **New Q&A pairs** — add entries to the `hvaerinnafor_qa_bank` index. Each
   node needs `answer`, `short_answer`, `severity`, `category`, `valid: 1`,
   and `from_doc_id` pointing to the source article.

3. **New categories** — add to the `categories` list in `answer_utils.py`.

4. **New topics** — update the severity classification prompt in
   `analyze_query` (`agent_workflow_answer.py`) and the category list in
   `registry.py` (`REFINE_AND_CLASSIFY` prompt).

---

## Deployment

The app is deployed as an **Azure App Service** (Linux, Python 3.12). On
startup it detects the Azure environment via the `WEBSITE_SITE_NAME` environment
variable and adjusts paths accordingly (`RunningLocally()` in `config.py`).

Vector indexes are mounted as a path from Azure Blob Storage at
`/blobstorage/chatbot/`.

The app listens on port 80 and is served directly by Quart's built-in ASGI
server in production. For higher throughput, consider fronting it with
**Hypercorn** (already in `requirements.txt`):

```bash
hypercorn app:app --bind 0.0.0.0:80 --workers 2
```
