# HEI 2.0 — KI-chatbot backend

Backend for the AI chatbot in **«Hva er innafor» 2.0 (HEI 2.0)**, developed by
Helsedirektoratet. The chatbot answers questions about relationships, sexuality,
feelings, and sexual health aimed at Norwegian youth aged 13–19.

---

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agent workflow](#agent-workflow)
- [Retrieval modes](#retrieval-modes)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Running locally](#running-locally)
- [API endpoints](#api-endpoints)
- [Session management](#session-management)
- [Severity classification](#severity-classification)
- [Stance routing and safety nodes](#stance-routing-and-safety-nodes)
- [Response styles](#response-styles)
- [Help-service catalogue](#help-service-catalogue)
- [Cost and token accounting](#cost-and-token-accounting)
- [Testing](#testing)
- [Adding new content](#adding-new-content)
- [Deployment](#deployment)

---

## Overview

The backend is a streaming HTTP API built with **Quart** (async Flask). It
exposes these endpoints to the HEI 2.0 frontend:

| Endpoint | Purpose |
|---|---|
| `POST /chat` (agent `hvaerinnafor`) | Main Q&A — returns a grounded, tone-styled answer |
| `POST /chat` (agent `hvaerinnafor_related_qa`) | Returns a pre-written answer for a question clicked in the related-questions list |
| `POST /examples` (agent `hvaerinnafor_examples`) | Returns categorised example questions for the landing page |
| `GET /healthz` | Readiness probe — always 200, body carries a `ready` flag |
| `GET /categories` | Distinct categories present in the article index |
| `GET /documents` | Distinct source documents in the article index |

Answers are streamed as **Server-Sent Events (SSE)** so the frontend can render
text progressively. The final answer is streamed token-by-token straight out of
the LLM, not buffered until the graph finishes.

---

## Architecture

```
Frontend
   │  SSE stream
   ▼
Quart HTTP server  (app.py / routes.py)
   │
   ├── GET  /healthz      index readiness (always 200, `ready` flag)
   ├── GET  /categories   distinct categories in the article index
   ├── GET  /documents    distinct source documents in the article index
   │
   ├── POST /chat  (agent: hvaerinnafor)
   │   answer_utils.get_answer_as_stream()
   │        │
   │        ▼
   │   LangGraph workflow  (agent_workflow_answer.py)
   │        │
   │   analyze_query          ONE fast_llm call:
   │        │                 refined query · severity · stance · tense ·
   │        │                 asker_gender · subqueries
   │        ▼
   │   route_after_analysis
   │    ├── respond_self_harm ──────────┐   no RAG, own LLM prompt
   │    ├── refuse_harm_to_others ──────┤   no RAG, own LLM prompt
   │    ├── help_after_harm ────────────┤   no RAG, own LLM prompt
   │    ├── address_prejudice ──────────┤   no RAG, own LLM prompt
   │    │                               │
   │    ├── fast_single ───────┐        │   one grounded call
   │    └── query_grounded ×N ─┤        │   Send fan-out, one per subquery
   │                           ▼        │
   │              synthesize_style_stream   merge sub-answers + apply tone,
   │                           │            streamed token-by-token
   │                           ▼        │
   │              emit_query_answer_references  ◄──────┘
   │                           │        query_status · references ·
   │                           │        short_answer · Token usage
   │                           ▼
   │              related_queries_dialog_from_query
   │
   ├── POST /chat  (agent: hvaerinnafor_related_qa)
   │   agent_workflow_qa.py  →  pre-written answer looked up by from_node_id
   │
   └── POST /examples
       answer_utils.get_examples_full_as_stream()
```

**Vector indexes** (LlamaIndex, persisted to Azure Blob Storage mount):

| Index name | Content |
|---|---|
| `hvaerinnafor` | Source articles — the factual knowledge base (Art mode) |
| `hvaerinnafor_qa_bank` | Pre-written Q&A pairs used for related-questions suggestions, the `/examples` cards, and the fast follow-up path |
| `hvaerinnafor_unified` | Single index merging article chunks **and** ung.no Q&A nodes into one retrieval pool (Hyb mode) |

The index the client asks for determines which **retrieval mode** is active —
see [Retrieval modes](#retrieval-modes) below.

---

## Agent workflow

The main `/chat` workflow (`agent_workflow_answer.py`) runs as a LangGraph
state machine over `State_Answer`. The compiled graph is exported to
[docs/graph.mmd](docs/graph.mmd) (with a Norwegian-labelled copy in
[docs/graph.no.mmd](docs/graph.no.mmd)); a route-by-route walkthrough in
Norwegian lives in [docs/answer_flow.md](docs/answer_flow.md).

```
START
  └─► analyze_query          ONE call on the fast model. Rewrites the query,
                              sets severity (Green/Yellow/Red), stance
                              (info_seeker / affected_party / harm_to_others /
                              harm_to_self / expresses_prejudice / ambiguous),
                              the tense when stance is harm_to_others
                              (planning / completed / unclear), asker_gender
                              (jente / gutt / ukjent), and — when the question
                              is compound — the subqueries themselves.
        ├─► respond_self_harm       stance=harm_to_self
        │                            Crisis support. Skips retrieval.
        ├─► refuse_harm_to_others   stance=harm_to_others + tense ∈ {planning, unclear}
        │                            Constructive refusal. Skips retrieval.
        ├─► help_after_harm         stance=harm_to_others + tense=completed
        │                            Damage-limitation guidance. Skips retrieval.
        ├─► address_prejudice       stance=expresses_prejudice
        │                            Meets the attitude without validating it.
        │                            Skips retrieval.
        ├─► fast_single             Single question — one grounded call
        └─► query_grounded ×N       Compound question — one worker per subquery,
                                     fanned out directly with Send
              └─► synthesize_style_stream   Merges the valid sub-answers AND
                                             applies the response style in one
                                             streamed LLM call
                    └─► emit_query_answer_references   query_status, references,
                          │                             short_answer, token usage
                          └─► related_queries_dialog_from_query   Follow-up suggestions
END
```

There is **no orchestrator node**: `analyze_query` produces the subqueries in
the same call that classifies the question, and `route_after_analysis` returns
a `Send` list that fans the workers out directly. That saves a full LLM
round-trip on every compound question.

`synthesize_style_stream` (`join=True`) replaces the former `synthesizer` +
`empathy_rewrite` / `apply_response_style` pair — merging and re-toning the
answer in one call instead of two, and streaming the result.

### Grounding and citation verification

Each `query_grounded` worker runs this pipeline for its subquery:

1. **Retrieve** nodes from the active index (`similarity_top_k` /
   `similarity_cutoff` from the request).
2. **Situational filter** — only when the question states a fact that some
   sources would contradict (see below).
3. **Relevancy band** from the best node score (`Strong` ≥ 0.60,
   `Medium` ≥ 0.55, otherwise `Rejected`). A `Rejected` band short-circuits the
   worker — no LLM call is made.
4. **Article floor** — `_ensure_article_in_top`
   ([agent_workflow_answer.py:596](agent_workflow_answer.py#L596)) guarantees at
   least one article node in the context when one qualifies (see
   [Hyb mode](#hyb-mode-unified)).
5. **Grounded answer** — `GROUNDED_PROMPT` produces a `GroundedAnswer`
   (answer + claims + citations) from at most `MAX_NODES_FOR_CONTEXT` (8) nodes,
   `MAX_CHARS_PER_NODE` (2 500) characters each. The prompt gets the severity
   and an `asker_gender` hint that steers the angle only — it never affects
   which nodes are retrieved, and `ukjent` keeps the answer gender-neutral.
6. **Citation check** — every quote is matched verbatim against the retrieved
   text, so invented quotes are caught.
7. **Entailment gate** — see below.
8. **Validity** — the sub-answer stays `valid` when the share of claims with at
   least one verified citation is ≥ `claims_valid_threshold` (default `1.0`).
   Sub-answers that end up `not valid` are dropped; if none survive, the turn
   returns a `CANNOT_ANSWER_PLACEHOLDER` and reports
   `validate_response_result: "Rejected"`.

### The entailment gate

A quote can appear verbatim in the source and still not support the claim
(a p-stav quote used for a p-plaster claim, say). The citation check only
catches invented quotes, so `_apply_entailment_gate` closes that hole:

- A cheap term-overlap pre-filter picks out the suspicious claims — most claims
  never reach the LLM.
- One batched call on the fast model judges those claims against their quotes
  **and** the surrounding source passage, so elliptical quotes ("Det er ulovlig
  for læreren") can be resolved instead of judged blind.
- Every claim the batch rejects is **re-checked alone** before it is downgraded.
  The batched call occasionally produces a flaky rejection on a borderline
  claim, and a strict `claims_valid_threshold` would then discard a perfectly
  grounded answer.
- The gate fails open: any error keeps the string-match verdict.

Turn it off per request with `entailment_check: false`.

### Situational node filter

Retrieval doesn't know what the user has already told us. Asked "jeg er gravid,
hva gjør jeg nå?", the index happily returns nodes about the morning-after pill
and pregnancy tests — advice for avoiding the exact situation the user is
already in.

`_filter_situational_nodes` drops those nodes. It only runs when
`_PREMISE_FACT_MARKERS` matches the question (an already-established fact such
as "er gravid", "testen var positiv", "har sendt bildet"), it is instructed to
be conservative, and it fails open — on any error, or if it would remove every
node, the original list is kept.

### Severity levels

| Level | Meaning | Example |
|---|---|---|
| **Green** | General information, preventive | "What is consent?" |
| **Yellow** | Difficult situation, needs empathy | "I regret sending a nude photo" |
| **Red** | Acute crisis, needs immediate help | "I was assaulted yesterday" |

Severity affects which related questions are shown, which response style is
picked, and which fallback placeholder is used when no answer is found.

---

## Retrieval modes

The `query_grounded` worker supports two retrieval modes. **There is no `mode`
flag** — the active mode is determined by which `vectorIndex` the client picks
and which indexes the server has loaded at startup.

| Mode (client pill) | `vectorIndex` | Behavior |
|---|---|---|
| **Art** | `hvaerinnafor` | Article-only index — every query goes straight to article chunks |
| **Hyb** (unified) | `hvaerinnafor_unified` | Unified index — articles and ung.no Q&A nodes compete in the same retrieval call |

The only difference between modes is which index they hit — that's the A/B
signal. The mapping above mirrors `RETRIEVAL_MODES` in the client
([HeiChatClient.jsx:80-83](https://github.com/cgoul-code/chatbot-client-HEI20-v2/blob/main/client/src/HeiChatClient.jsx#L80-L83)) — see the
client README for example queries that exercise each mode.

### Art mode

Only the article index is queried for grounding. The `hvaerinnafor_qa_bank`
index is still loaded but is used purely for the related-questions suggestion
list and for the fast follow-up path (`agent: hvaerinnafor_related_qa`).

### Hyb mode (unified)

A single index (`hvaerinnafor_unified`) holds two physically distinct kinds of
nodes that compete in the same retrieval call:

- **Article chunks** — `answer-text` = article prose, `aliases` = 10 LLM-
  generated synthetic questions per chunk **plus up to 3 cross-pollinated real
  ung.no questions** (top-3 matches with similarity ≥ 0.55).
- **Q&A nodes** — `answer-text` = the original ung.no veileder's answer,
  `aliases` = the original user question.

The `aliases` field is **embedding-only**: it shapes which node gets retrieved
but never enters the LLM prompt. The LLM only ever sees the winning nodes'
`answer-text`.

**Cross-pollination** is a retrieval-steering trick at build time. By grafting
real user questions onto the article chunks they best match, articles become
competitive when the user phrases their query the way a real ung.no user did —
even if the article's own prose doesn't sound like a question. It does **not**
merge the two node types; it just makes article chunks "speak user."

At query time the top-K can return all-article, all-Q&A, or a mix — whichever
embeddings rank highest.

**Article-floor guarantee.** After retrieval, `query_grounded` runs
`_ensure_article_in_top` ([agent_workflow_answer.py:596](agent_workflow_answer.py#L596))
on the result before truncating to the LLM context. If the score-ordered top-N
contains no `node_type=="article"` node but at least one article appears
further down the retriever output (i.e. it cleared `similarity_cutoff`), the
lowest-ranked `node_type=="qa"` node in the top-N is swapped out for the
highest-ranked article. This keeps the grounding set from collapsing to
Q&A-only on queries where article phrasing happens to lose the embedding race,
while preserving the highest-scoring Q&A signal. If no article qualifies the
behavior is unchanged (pure score order); in Art mode the rule is a no-op
since every node is an article. A `systeminfo` event
`"Article promoted into top-N …"` is emitted when the swap actually fires.

### Choosing a mode

| If you want… | Use |
|---|---|
| Article-only baseline (original behavior) | Art |
| Articles + ung.no Q&A ranked together in one retrieval call | Hyb |

---

## Project structure

```
.
├── app.py                        Quart app entry point
├── routes.py                     HTTP route handlers + session management
├── config.py                     Server settings, index registry, index loading
├── llm_provider.py               Chat-model factory (main + fast model, per LLM_PROVIDER)
├── embeddings_provider.py        Embedding-model factory
├── query_utils.py                QuerySettings — request-body parsing
├── answer_utils.py               Agent entry points (stream wrappers) + /examples cache
├── agent_workflow_answer.py      Main LangGraph Q&A workflow
├── agent_workflow_qa.py          Related-question lookup workflow
├── agent_shared.py               Shared helpers (emit, normalize, retriever builder)
├── registry.py                   Prompt templates + static safety answers
├── hjelpetjenester_ungdom.json   Help-service catalogue used by the safety nodes
├── json_utils.py                 Safe JSON parsing for LLM outputs
├── graph_utils.py                Mermaid diagram export helper
├── generate_platform_doc.py      Regenerates Teknisk_plattform_HEI_chatbot.docx
├── docs/                         Flow documentation and exported graphs
├── test/                         Fixtures, classification test runner, smoke scripts
├── requirements.txt
└── .env                          Environment variables (not committed)
```

See [README_UTILS.MD](README_UTILS.MD) for day-to-day developer and Azure
operations commands.

---

## Prerequisites

- Python 3.11+ (CI builds and deploys on 3.12)
- Access to **Azure OpenAI** (chat completion + embeddings deployments), or an
  Anthropic / Mistral key if you switch `LLM_PROVIDER`
- Vector indexes built and available at the paths in `VECTOR_INDEX_MAP` (see
  `config.py`), either locally or mounted from Azure Blob Storage

---

## Configuration

Create a `.env` in the project root:

```env
# ── Chat model ────────────────────────────────────────────────────────────
# azure_openai (default) | anthropic | mistral
LLM_PROVIDER=azure_openai

AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-mini
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/

# Optional cheaper/faster deployment for the auxiliary calls (query analysis,
# entailment gate, situational filter, related-query selection). Falls back to
# the main deployment when unset.
AZURE_OPENAI_FAST_DEPLOYMENT_NAME=gpt-4o-mini

# ── Embeddings ────────────────────────────────────────────────────────────
AZURE_OPENAI_EMBEDDINGS_MODEL=text-embedding-3-large
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_EMBEDDINGS_API_KEY=<your-key>
AZURE_OPENAI_EMBEDDINGS_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_EMBEDDINGS_API_VERSION=2024-02-01

# ── Cost tracking (USD per 1 million tokens) ──────────────────────────────
PRICE_INPUT_USD_PER_M=0.40
PRICE_OUTPUT_USD_PER_M=1.60
# Fast model — falls back to the main-model price when unset
PRICE_FAST_INPUT_USD_PER_M=0.15
PRICE_FAST_OUTPUT_USD_PER_M=0.60
# Optional; defaults to 10.0
USD_TO_NOK=10.5

# ── Blob storage (index download) ─────────────────────────────────────────
CONNECTION_STRING=<storage-connection-string>
CONTAINER_NAME=chatbot
INDEX_BLOB_PREFIX=<prefix>
```

Switching provider replaces the chat model only — embeddings stay on Azure
OpenAI, since the indexes were built with them:

| `LLM_PROVIDER` | Main model | Fast model (optional) |
|---|---|---|
| `azure_openai` | `AZURE_OPENAI_DEPLOYMENT_NAME` | `AZURE_OPENAI_FAST_DEPLOYMENT_NAME` |
| `anthropic` | `ANTHROPIC_MODEL` | `ANTHROPIC_FAST_MODEL` |
| `mistral` | `MISTRAL_MODEL` | `MISTRAL_FAST_MODEL` |

The provider SDKs are imported lazily, so only the one you select has to be
installed.

---

## Running locally

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env with your credentials (see Configuration above)

# 4. Place vector indexes at the expected paths (or symlink them):
#    ./blobstorage/chatbot/hvaerinnafor
#    ./blobstorage/chatbot/hvaerinnafor_qa_bank
#    ./blobstorage/chatbot/hvaerinnafor_unified

# 5. Start the server
python app.py
# Server starts at http://0.0.0.0:80
```

The server loads indexes asynchronously after startup. `/chat` and `/examples`
return HTTP 503 until they are ready; poll `GET /healthz` (always 200) and wait
for `"ready": true`.

---

## API endpoints

### `POST /chat`

Streams an answer to a user question. The shape below matches what the HEI 2.0
client actually sends from
[HeiChatClient.jsx](https://github.com/cgoul-code/chatbot-client-HEI20-v2/blob/main/client/src/HeiChatClient.jsx)
— shared fields are identical across modes; only `vectorIndex` changes.

**Headers:**
```
Content-Type: application/json
Cache-Control: no-cache
Accept: text/event-stream
```

**Request fields** (parsed by `get_query_settings` in
[query_utils.py](query_utils.py); every field is optional except `messages`):

| Field | Type | Default / typical value |
|---|---|---|
| `messages` | array | `[{ role: 'user', content: '<spørsmål>' }]` — the current user input. The server keeps its own history; only the last user message is read from the payload |
| `session_id` | string | uuid v4 generated per page load. Created server-side and returned in the `open` event when omitted |
| `clientType` | string | `"HEI"` |
| `agent` | string | `"hvaerinnafor"`, or `"hvaerinnafor_related_qa"` for related-question follow-ups |
| `similarity_top_k` | string \| number | `10`. The client sends `"5"` |
| `similarity_cutoff` | string \| number | `0.45`. The client sends `"0.75"` |
| `relevancy_cutoff` | number | `0.45` |
| `response_mode` | string | `"tree_summarize"` — LlamaIndex response-synthesizer mode |
| `stream` | boolean | `true` |
| `from_related_q` | boolean | `true` if the user clicked a related-question suggestion |
| `from_node_id` | string \| null | source node id when `from_related_q` is `true` |
| **`vectorIndex`** | string | **per [retrieval mode](#retrieval-modes)** |
| `qa_bank_index` | string \| null | Explicit QA-bank index name (related-question suggestions, `/examples` cards and the fast follow-up path). If omitted or the named index isn't loaded, the server tries `{vectorIndex}_qa_bank` and then falls back to `hvaerinnafor_qa_bank`. Set it to silence the "QA-bank … not found" warning when running a `vectorIndex` with no matching `_qa_bank` (e.g. `hvaerinnafor_unified` → `"hvaerinnafor_qa_bank"`) |
| `response_style` | string \| null | Override for the answer style: `"factual"` (skip rewrite), `"warm"`, `"supportive"`, `"crisis"`. Omit or send an unknown value to auto-route from severity + stance. See [Response styles](#response-styles) |
| `claims_valid_threshold` | number | `1.0` — minimum share of claims that must carry a verified citation for a sub-answer to stay valid |
| `entailment_check` | boolean | `true` — run the [entailment gate](#the-entailment-gate) |

`debug_emit_nodes` also exists on `QuerySettings` (it makes `query_grounded`
emit the exact retrieved nodes as a `retrieved_nodes` event), but it is set
programmatically by the test harness — it is deliberately not read from the
request body.

**Per-mode values:**

| Mode | `vectorIndex` |
|---|---|
| Art | `"hvaerinnafor"` |
| Hyb (unified) | `"hvaerinnafor_unified"` |

**Example payload — Hyb mode:**
```json
{
  "messages": [{ "role": "user", "content": "Kan jeg slutte på videregående?" }],
  "session_id": "f7c1e9...",
  "clientType": "HEI",
  "agent": "hvaerinnafor",
  "similarity_top_k": "5",
  "similarity_cutoff": "0.75",
  "vectorIndex": "hvaerinnafor_unified",
  "qa_bank_index": "hvaerinnafor_qa_bank",
  "stream": true,
  "from_related_q": false,
  "from_node_id": null
}
```

Swap `vectorIndex` to `"hvaerinnafor"` to reproduce Art behavior. The client
README has full worked examples per mode.

**SSE events emitted:**

| Event name | Content |
|---|---|
| `open` | First event of every stream. Carries the resolved `session_id` |
| `answer` | Answer text delta (markdown). Streamed token-by-token from `synthesize_style_stream`; emitted line-by-line for the safety routes and placeholders |
| `short_answer` | One-sentence summary |
| `references` | `[Title](url) \|\|IMG\|\| icon_url` per source |
| `related queries` | JSON array of follow-up question suggestions |
| `Refined query` | Rewritten version of the user's question |
| `query_status` | JSON snapshot of the turn's classification, retrieval state and cost — see [query_status payload](#query_status-payload). Emitted **twice** |
| `info` | Internal pipeline step names (for debugging) |
| `systeminfo` | Claim/citation verification details (for debugging) |
| `retrieved_nodes` | Debug only: the exact retrieved nodes (score, url, node_type, text). Off unless `debug_emit_nodes` is set |
| `Token usage` | Token counts and estimated NOK cost |
| `done` | Stream completed |
| `error` | Error message if something failed |

The stream also carries a `: keepalive` SSE comment every 15 seconds so idle
proxies don't close the connection.

To use the **related-question fast path**, set `agent` to
`hvaerinnafor_related_qa` and include `from_node_id` (the node ID from a
previously emitted `related queries` item).

#### `query_status` payload

Emitted **twice** per turn, both times as a single JSON object on the event:

1. from `emit_query_answer_references`, with the cost of everything up to the
   answer, and
2. again from `related_queries_dialog_from_query`, folding in the cost of the
   related-questions call, which runs after the first emit.

Use the last one you receive for cost; the classification fields are identical
in both. It fires for **all** terminal paths — including the safety branches
(`respond_self_harm`, `refuse_harm_to_others`, `help_after_harm`,
`address_prejudice`), since they all join back into
`emit_query_answer_references` before END. That makes it the single reliable
signal for how a turn was classified.

```json
{
  "refined_query": "Hvordan virker p-piller?",
  "query_severity": "Green",
  "stance": "info_seeker",
  "harm_to_others_tense": "na",
  "asker_gender": "ukjent",
  "response_style": "factual",
  "response_style_source": "auto",
  "relevancy_band": "Strong",
  "best_node_score": 0.812,
  "validate_response_result": "Accepted",
  "input_tokens": 5321,
  "output_tokens": 742,
  "main_input_tokens": 4180,
  "main_output_tokens": 631,
  "fast_input_tokens": 1141,
  "fast_output_tokens": 111,
  "cost_usd": 0.0031,
  "cost_nok": 0.0326
}
```

| Field | Source | Notes |
|---|---|---|
| `refined_query` | `analyze_query` | Rewritten search-friendly form of the user's question |
| `query_severity` | `analyze_query` | `"Green"` / `"Yellow"` / `"Red"` / `""` |
| `stance` | `analyze_query` | `"info_seeker"` / `"affected_party"` / `"harm_to_others"` / `"harm_to_self"` / `"expresses_prejudice"` / `"ambiguous"` |
| `harm_to_others_tense` | `analyze_query` | `"planning"` / `"completed"` / `"unclear"` / `"na"`. Normalised to `"na"` whenever stance ≠ `harm_to_others` |
| `asker_gender` | `analyze_query` | `"jente"` / `"gutt"` / `"ukjent"`. Only set when the user's own words make it explicit; steers the answer's angle, never retrieval |
| `response_style` | `synthesize_style_stream` | Final effective style — `"factual"` / `"warm"` / `"supportive"` / `"crisis"`. Empty string `""` when the styling step was bypassed (safety routes, Rejected turns) |
| `response_style_source` | `synthesize_style_stream` | How the style was picked: `"auto"`, `"override"` (client sent a valid `response_style`), `"forced_red"` (severity=Red safety floor), or `""` (step skipped) |
| `relevancy_band` | derived in `emit_query_answer_references` | `"Strong"` / `"Medium"` / `"Rejected"` — from the best retrieval score across completed subqueries. Empty `""` when no retrieval ran |
| `best_node_score` | derived in `emit_query_answer_references` | `max(response_validity_index)` over the completed subqueries. `0.0` when no retrieval ran |
| `validate_response_result` | `synthesize_style_stream` / `fast_single` / safety nodes | `"Accepted"` when an answer was produced, `"Rejected"` when it is a `CANNOT_ANSWER_PLACEHOLDER` |
| `input_tokens` / `output_tokens` | `_compute_cost` | Totals across both models |
| `main_*` / `fast_*` tokens | `_compute_cost` | Split per model, so the cost is reproducible — the two are priced separately |
| `cost_usd` / `cost_nok` | `_compute_cost` | See [Cost and token accounting](#cost-and-token-accounting) |

Typical use on the client: surface these fields behind a debug info-icon on
each turn, classify test fixtures by stance/tense, or filter production logs
for `validate_response_result == "Rejected"`. The HEI 2.0 frontend wires this
into a per-message popover; see the `case 'query_status':` handler in
[HeiChatClient.jsx](https://github.com/cgoul-code/chatbot-client-HEI20-v2/blob/main/client/src/HeiChatClient.jsx).

On the safety routes:

- `stance` is `"harm_to_self"`, `"harm_to_others"` or `"expresses_prejudice"`;
  `harm_to_others_tense` is only meaningful for `harm_to_others`.
- `validate_response_result` is `"Accepted"` — the safety nodes write their own
  text and don't go through claim verification.
- `response_style` / `response_style_source` are both `""`, since
  `synthesize_style_stream` is bypassed.
- `relevancy_band` is `""` and `best_node_score` is `0.0` — no retrieval ran.

### `POST /examples`

Returns categorised example questions for the landing page. No streaming —
emits a single `examples_categories` event followed by `done`. The pools are
built once per QA-bank index and cached in-process.

**Request body:**
```json
{
  "agent": "hvaerinnafor_examples",
  "vectorIndex": "hvaerinnafor",
  "qa_bank_index": "hvaerinnafor_qa_bank",
  "requested_categories": ["Forelskelse / Flørting", "Sex / Intimitet"]
}
```

Omit `requested_categories` to get every category the QA-bank has.

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

### `GET /healthz`

Readiness probe. Always returns **200** — the body says whether the indexes
have finished loading, which keeps the client's polling loop simple. `/chat`
and `/examples` still return 503 while loading.

```json
{ "ready": false, "status": "Loading indexes...", "message": "Serveren laster fortsatt indekser. Prøv igjen om noen sekunder." }
```

### `GET /categories`

Returns the sorted distinct `categories` values found in the `hvaerinnafor`
index — a JSON array of strings.

### `GET /documents`

Returns the distinct source documents in the `hvaerinnafor` index, deduplicated
by URL and sorted by title:

```json
{
  "count": 128,
  "documents": [
    { "url": "https://...", "title": "...", "category": "...", "categories": ["..."], "description": "..." }
  ]
}
```

---

## Session management

Conversation history is stored server-side in a **diskcache** store
(`./session_cache`), keyed by `session_id`. The client receives the
`session_id` in the `open` event and should send it back on subsequent
requests; if it doesn't, the server generates a new one and the turn starts
without history.

Only the *last* user message is read from the request body — the rest of the
history comes from the server-side store, so a client can't rewrite the past.

History is automatically trimmed to stay within:
- **20 messages** maximum
- **8 000 characters** total

Trimming always preserves complete user/assistant pairs and ensures history
starts with a user message.

If the client disconnects mid-stream, the workflow is cancelled and the partial
assistant message is still written to the session, so the next turn sees what
the user actually saw.

---

## Severity classification

The `analyze_query` node classifies every incoming question into one of three
severity levels before retrieval. This affects:

- Which related questions are surfaced (Green questions only see Green
  candidates; Yellow sees Green + Yellow; Red sees all)
- Which [response style](#response-styles) is used — `Red` is always clamped to
  `crisis`
- Which fallback placeholder is shown when no answer is found

Questions expressing thoughts of or intent to self-harm are always classified
`Red`. For Red questions the fallback always directs the user to professional
help resources.

---

## Stance routing and safety nodes

`analyze_query` doesn't only assess severity — it also classifies the user's
**stance**. This is a separate axis from severity and it decides whether the
question should be answered from the knowledge base at all, or routed to a
dedicated safety node.

### Stance

| Stance | Meaning | Example | Route |
|---|---|---|---|
| `info_seeker` | Asks for general knowledge or advice, no specific actor | "Hva sier loven om deling av nakenbilder?" | normal RAG |
| `affected_party` | Describes something affecting the user themselves (victim, pressured, worried) | "Eksen min truer med å dele nakenbilder av meg" | normal RAG |
| `harm_to_others` | Asks for help performing an illegal/harmful act on another person | "Jeg vil dele nakenbilder av eksen min" | `refuse_harm_to_others` / `help_after_harm` |
| `harm_to_self` | Expresses thoughts of, or intent to, self-harm or suicide | "Jeg orker ikke mer" | `respond_self_harm` |
| `expresses_prejudice` | Expresses a prejudice or demeaning attitude towards a group | "Jeg liker ikke homofile" | `address_prejudice` |
| `ambiguous` | Unclear who is the actor and who is the target | — | normal RAG |

The classifier is told to break ties toward `harm_to_others` when a concrete
intention is expressed ("jeg vil …", "hva er enkleste måten å …"), and toward
`info_seeker` for pure knowledge questions ("hva sier loven om …").

`harm_to_self` wins over every other stance — it is checked first in
`route_after_analysis`.

### Tense (only when stance = `harm_to_others`)

When stance is `harm_to_others`, a second classification picks the **tense** of
the harmful action. This decides which safety node handles the response:

| Tense | Triggers on | Routes to |
|---|---|---|
| `planning` | Future/modal verbs: "vil", "skal", "hvordan kan jeg", "hva er enkleste måten å …" | `refuse_harm_to_others` |
| `completed` | Past tense / present perfect: "har sendt", "delte", "filmet", "installerte" | `help_after_harm` |
| `unclear` | Ambiguous (e.g. "har bildene og vurderer å dele dem") | `refuse_harm_to_others` |
| `na` | Stance is not `harm_to_others` | (normal flow) |

`analyze_query` normalises the tense to `na` for every other stance, so routing,
the UI and the test fixtures never see a tense that doesn't apply.

### Safety nodes

All four safety nodes are **LLM-driven** with strict system prompts in
[registry.py](registry.py), and fall back to a hardcoded message if the LLM call
fails:

| Node | Prompt | Static fallback |
|---|---|---|
| `refuse_harm_to_others` | `REFUSE_HARM_PROMPT` | `HARM_REFUSAL_ANSWER` |
| `help_after_harm` | `HELP_AFTER_HARM_PROMPT` | `HELP_AFTER_HARM_ANSWER` |
| `respond_self_harm` | `SELF_HARM_PROMPT` | `SELF_HARM_ANSWER` |
| `address_prejudice` | `ADDRESS_PREJUDICE_PROMPT` | `PREJUDICE_ANSWER` |

They all **skip retrieval and the styling step**:

- Retrieval is skipped because the knowledge base is written for victims and
  general info — answering a perpetrator, a self-harm disclosure or a prejudiced
  statement from the victim corpus produces the wrong frame. For prejudice in
  particular, retrieval tends to return "accept yourself" articles and
  misreads the user as struggling with their own identity.
- `synthesize_style_stream` is skipped because the style prompts are written to
  re-tone a *grounded* answer; a warm rewrite on top of a refusal would soften
  it inappropriately. These nodes produce already-styled text.

**`refuse_harm_to_others`** — for planning/unclear cases. Acknowledges the
underlying feeling briefly, states clearly that the act is illegal or harmful
(the LLM cites Norwegian penal code when it can be precise: §267a image sharing,
§282 intimate partner abuse, §263/266 threats, §297–299 sexual assault,
§305 grooming, §312 incest, §145 stalkerware, etc.), **never describes how the
act could be performed**, and points to 2–3 concrete help services.

**`help_after_harm`** — for completed cases. **Not** a refusal — the harm is
already done, and a cold refusal helps no one. Acknowledges that reaching out
takes courage, gives a realistic picture of the legal and personal consequences,
and provides concrete next steps focused on damage limitation: stop further
spread, talk to a trusted adult, consider contacting the police or a lawyer
proactively, limit harm for the affected party, and get help for the user
themselves.

**`respond_self_harm`** — a warm crisis response that acknowledges the user,
conveys hope, and guides them on to crisis help. Mental Helse Ungdom (116 123,
open around the clock) is injected deterministically if the LLM leaves it out.

**`address_prejudice`** — meets the discomfort without validating the
prejudice, and states plainly that other people are of equal worth and have the
right to be who they are.

### Deterministic service guarantees

The LLM picks help services by instruction, and instructions can miss. A few
situations are specific enough that the right service must never fall out of a
harm answer, so `_inject_specialized_harm_services` adds it back by regex:

| Signal in the question/history | Service guaranteed |
|---|---|
| The user's own sexual boundary violation (contact/behaviour) | Tryggprat |
| Images, video or private information involved | Slettmeg.no |
| Ongoing or immediate danger right now | Politiet (akutt) / 112 |
| Self-harm route | Mental Helse Ungdom (116 123) |

---

## Response styles

`synthesize_style_stream` merges the valid sub-answers and rewrites them to fit
the situation — in one streamed LLM call. Instead of one prompt with
severity-conditional hints, there are **four named profiles**, each with its
own prompt and its own examples, so no profile leaks the wrong tone into
another.

| Style | When used | Characteristics |
|---|---|---|
| `factual` | `info_seeker` / `ambiguous` + Green | **No re-toning.** With a single valid sub-answer the raw text is streamed straight out — no LLM call at all. With several sub-answers a neutral faithful-merge prompt joins them |
| `warm` | `info_seeker` + Yellow, or `affected_party` + Green | Light normalization, at most one acknowledging sentence, then facts. Conversational but not heavy |
| `supportive` | `affected_party` + Yellow | Opens with validation ("det du beskriver høres vanskelig ut"), then facts, closes with presence ("du står ikke alene") |
| `crisis` | Severity = Red, any stance | Direct support first, single clearest next action, short and focused. Avoids long bullet menus |

### Client override

The client can override the auto-routing by sending `response_style` in the
`POST /chat` body, with one of `"factual"`, `"warm"`, `"supportive"`,
`"crisis"`. Omit the field (or send an unknown value) to use auto-routing.

**Precedence:**
1. **Safety floor:** if `severity == "Red"`, the style is always clamped to
   `crisis` — *even if the client requested something else*. A client override
   of e.g. `factual` on an acute crisis would produce a cold, fact-only
   response on a Red question, which is never appropriate. The floor only
   matters when an override is present (auto-routing returns `crisis` on Red
   anyway).
2. Valid `response_style` in the request body → that style wins (subject to the
   safety floor above).
3. Field absent, empty, or unknown → server falls through to
   `pick_response_style(severity, stance)`.

Use overrides for A/B testing tone, for letting users pick their own
preference, or for forcing a particular style during evaluation.

```json
{
  "messages": [{ "role": "user", "content": "..." }],
  "session_id": "...",
  "vectorIndex": "hvaerinnafor_unified",
  "response_style": "warm"
}
```

**Client UI (HEI 2.0):** the chatbot client exposes a pill row mirroring this —
`Auto / Fakta / Vennlig / Støtte / Krise`. "Auto" sends nothing (field omitted
via spread, so the server auto-routes). The other pills map 1:1 to the four
style values. The style is read per request; already-rendered turns keep
whatever style they were sent under. See
[`RESPONSE_STYLES` in HeiChatClient.jsx](https://github.com/cgoul-code/chatbot-client-HEI20-v2/blob/main/client/src/HeiChatClient.jsx).

**Debugging:** `synthesize_style_stream` emits an `info` event before it starts
streaming:

```
Synthesize + style (supportive, source=auto) — streaming
```

`source` is `auto`, `override` or `forced_red`, which makes it easy to see in
the stream whether a surprising tone came from auto-routing, an unexpected
client value, or the safety floor. The same values end up in `query_status` as
`response_style` / `response_style_source`.

**Unknown values are safe:** if the client sends a typo or a style we've
removed (`response_style: "casual"`), the server doesn't crash — the value
fails the `in RESPONSE_STYLES` check and auto-routing takes over.

### Why the styling step exists at all

The grounded answer from `query_grounded` is fact-correct but often tonally
neutral. Re-toning it lets us vary tone independently from retrieval — same
facts, different presentation. Merging and styling in the same call keeps that
to one LLM round-trip, and the `factual` single-answer path skips even that.

### Skipped paths

The four safety nodes (`respond_self_harm`, `refuse_harm_to_others`,
`help_after_harm`, `address_prejudice`) bypass `synthesize_style_stream`
entirely — they produce already-styled text from their own prompts. Their
answers are emitted (unstreamed) in `emit_query_answer_references`, which
checks the `answer_streamed` flag so nothing is written twice.

---

## Help-service catalogue

The safety nodes don't invent phone numbers. They pick from a single catalogue
in [hjelpetjenester_ungdom.json](hjelpetjenester_ungdom.json), loaded by
[registry.py](registry.py) and formatted into the prompts as
`HJELPETJENESTER_KATALOG`:

`mental-helse-ungdom`, `kors-paa-halsen`, `politiet-ikke-akutt`,
`politiet-akutt`, `sex-og-samfunn`, `snakk-med-noen-du-stoler-paa`, `slettmeg`,
`amathea`, `alarmtelefonen-barn-unge`, `tryggprat`,
`ungdomstelefonen-skeiv-ungdom`.

To change a phone number, opening hours or blurb, edit the JSON — every prompt,
every static fallback answer, and the deterministic injection rules read from
the same file. `HJELPETJENESTER_BY_ID` gives the lookup used by
`_ensure_service_in_answer`.

---

## Cost and token accounting

Two models run in one turn, and they are priced separately:

| Bucket | Calls |
|---|---|
| **Main** (`llm`) — `input_tokens` / `output_tokens` | Grounded answers, the safety nodes, `synthesize_style_stream` |
| **Fast** (`fast_llm`) — `fast_input_tokens` / `fast_output_tokens` | `analyze_query`, the entailment gate, the situational filter, related-query selection |

Both are `Annotated[int, add]` reducers in `State_Answer`, so each node returns
only its **delta**, never `state + delta` — returning the running total would
double-count.

`_compute_cost` prices each bucket with its own env variables
(`PRICE_*_USD_PER_M` for main, `PRICE_FAST_*_USD_PER_M` for fast, falling back
to the main price when unset) and converts with `USD_TO_NOK` (default `10.0`).
A missing variable prices at 0 rather than crashing.

The result reaches the client twice: on the `Token usage` event, and as the
token/cost fields of [`query_status`](#query_status-payload). The second
`query_status` — emitted from the related-questions node — is the one that
includes every call in the turn.

`test/_cost_smoke.py` prints both `query_status` events and the final
breakdown for one question, which is the quickest way to check the accounting
after touching a node.

---

## Testing

The test harness lives in [test/](test/) and drives the **full** agent (not
mocks), reading each turn's `query_status` to decide pass/fail.

```powershell
.\.venv\Scripts\Activate.ps1
python -m test.run_classification_tests          # interactive fixture menu
```

Fixtures are JSON lists of questions in `test/data/`. Each item declares its
own checks:

| Key | Check |
|---|---|
| `stance` | `"{stance} - {tense}"` must match, e.g. `"harm_to_others - planning"` |
| `rejected` | `validate_response_result == "Rejected"` must match the boolean |
| `expected_behaviour` | Free-text description of how the answer should behave; an LLM judge decides whether the actual answer satisfies it |

If several keys are present, all must pass. Results are written per fixture to
`test/results/<fixture>.xlsx`, plus `summary.xlsx` and `failures.xlsx`.

Non-interactive runs and A/B switches:

```powershell
python test/run_classification_tests.py `
    --files harm_to_self_questions.json,infoseeker_expected_behaviour_questions.json `
    --vector-index hvaerinnafor_unified `
    --topk 5 --cutoff 0.75 `
    --claims-threshold 1.0

# A/B the entailment gate
python test/run_classification_tests.py --files infoseeker_questions.json --no-entailment
```

Two ad-hoc smoke scripts (not fixtures) print `query_status` and the answer for
a handful of questions:

```powershell
python test/_cost_smoke.py      # per-model token/cost accounting
python test/_gender_smoke.py    # asker_gender handling
```

To dump answers for a question list into Excel, see
`test/data/batch_answer_to_excel.py` (examples in
[README_UTILS.MD](README_UTILS.MD)).

---

## Adding new content

1. **New articles** — add source documents to the `hvaerinnafor` index using
   the existing ingestion pipeline. Each document needs `title`, `url`, and
   optionally `icon_url` in its metadata. For unified mode, the same articles
   must also be re-indexed into `hvaerinnafor_unified` so the synthetic-question
   aliases and the cross-pollination step run on them.

2. **New Q&A pairs** — add entries to the `hvaerinnafor_qa_bank` index. Each
   node needs `answer`, `short_answer`, `severity`, `category`, `valid: 1`,
   and `from_doc_id` pointing to the source article.

3. **New ung.no Q&A entries** — re-run the unified-index build so the entries
   are ingested as Q&A nodes and considered for cross-pollination onto article
   chunks.

4. **New categories** — add to the `categories` list in `answer_utils.py`.

5. **New topics** — update the prompt and severity examples in
   `ANALYZE_QUERY_PROMPT` ([registry.py](registry.py)) and the category list
   passed in from `answer_utils.py`.

6. **New or changed help services** — edit
   [hjelpetjenester_ungdom.json](hjelpetjenester_ungdom.json). Add a
   deterministic guarantee in `_inject_specialized_harm_services`
   ([agent_workflow_answer.py](agent_workflow_answer.py)) only when a service
   must never be missing from a particular kind of answer.

7. **New workflow node** — after changing the graph, re-export the diagrams
   with `save_mermaid_diagram(answer_workflow.get_graph())`
   ([graph_utils.py](graph_utils.py); the call is commented out at the bottom of
   `agent_workflow_answer.py`) and update
   [docs/answer_flow.md](docs/answer_flow.md).

---

## Deployment

The app is deployed as an **Azure App Service** (Linux, Python 3.12). On
startup it detects the Azure environment via the `WEBSITE_SITE_NAME`
environment variable and adjusts paths accordingly (`RunningLocally()` in
`config.py`).

Vector indexes are mounted as a path from Azure Blob Storage at
`/blobstorage/chatbot/`.

Deployment is automated by the GitHub Actions workflow
[.github/workflows/main_helsesvar-chatbot-server-mini.yml](.github/workflows/main_helsesvar-chatbot-server-mini.yml):
every push to `main` builds on Python 3.12, zips the repo and deploys to the
`helsesvar-chatbot-server-mini` web app.

The App Service startup command is `bash startup.sh`, which creates/activates
the `antenv` virtualenv, installs `requirements.txt` and serves the app with
**Hypercorn**:

```bash
python -m hypercorn app:app --bind 0.0.0.0:${PORT:-8000}
```

Locally, `python app.py` runs Quart's built-in server on port 80 instead.

Operational commands (log tailing, SSH into the web app, downloading index
containers) are collected in [README_UTILS.MD](README_UTILS.MD).
