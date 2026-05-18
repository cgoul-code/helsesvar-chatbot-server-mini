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
- [Safety routing (harm_to_others)](#safety-routing-harm_to_others)
- [Response styles](#response-styles)
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
   │    analyze_query   (rewrite + severity + stance + tense)   │
   │         │                                                  │
   │    ┌────┼─────┬───────────────┬──────────────────┐         │
   │    │    │     │               │                  │         │
   │ fast_  orch  refuse_harm_   help_after_harm                │
   │ single estrator to_others   (stance=harm_to_                │
   │    │    │   (stance=harm_   others, tense=                  │
   │    │    │    to_others,    completed)                       │
   │    │    │    tense≠completed)                              │
   │    │    │                                                   │
   │    │    │      query_grounded  ×N  (parallel workers)       │
   │    │    │            │                                      │
   │    │    │      synthesizer                                  │
   │    │    └────────────┘                                      │
   │    │         │                                              │
   │    empathy_rewrite     ◄── refuse / help-after-harm SKIP    │
   │         │                  empathy_rewrite (deres tekst er  │
   │         │                  allerede formulert)              │
   │    emit_query_answer_references                             │
   │         │                                                   │
   │    related_queries_dialog_from_query                        │
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
| `hvaerinnafor` | Source articles — the factual knowledge base (legacy / hybrid mode) |
| `hvaerinnafor_qa_bank` | Pre-written Q&A pairs used for related-questions suggestions and the fast follow-up path |
| `psa_ssa_topics` | ~7 900 real Q&A entries from ung.no (last 12 months). Optional tier-1 short-circuit when loaded |
| `hvaerinnafor_unified` | Single index merging article chunks **and** ung.no Q&A nodes into one retrieval pool (unified mode) |

The indexes loaded at startup determine which **retrieval mode** is active — see
[Retrieval modes](#retrieval-modes) below.

---

## Agent workflow

The main `/chat` workflow (`agent_workflow_answer.py`) runs as a LangGraph state
machine:

```
START
  └─► analyze_query          Rewrites query, detects severity (Green/Yellow/Red),
                              classifies stance (info_seeker / affected_party /
                              harm_to_others / ambiguous) and — if harm_to_others
                              — the tense (planning / completed / unclear),
                              decides if subqueries are needed
        ├─► refuse_harm_to_others   stance=harm_to_others + tense ∈ {planning, unclear}
        │                            LLM-driven constructive refusal. Skips
        │                            retrieval and empathy_rewrite.
        ├─► help_after_harm         stance=harm_to_others + tense=completed
        │                            LLM-driven damage-limitation guidance.
        │                            Skips retrieval and empathy_rewrite.
        ├─► fast_single             Single question — one worker, direct answer
        └─► orchestrator            Multi-topic — splits into subqueries
              └─► query_grounded  ×N   Retrieval → grounded answer → citation check
                    └─► synthesizer   Merges sub-answers into one coherent response
                          └─► apply_response_style   Rewrites tone per response-style profile
                                                      (factual / warm / supportive / crisis).
                                                      'factual' skips the rewrite entirely.
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

## Retrieval modes

The `query_grounded` worker supports three retrieval modes. **There is no `mode`
flag** — the active mode is determined by which `vectorIndex` the client picks
and which indexes the server has loaded at startup.

| Mode (client pill) | `vectorIndex` | `psa_ssa_threshold` | Behavior |
|---|---|---|---|
| **Legacy** (`FAQ->Art`) | `hvaerinnafor` | `0.65` | Steg-1 Q&A shortcut on; fallback to article-only index |
| **Hybrid** (`FAQ->Hyb`) | `hvaerinnafor_unified` | `0.65` | Steg-1 Q&A shortcut on; fallback to unified index |
| **Unified** (`Hyb`) | `hvaerinnafor_unified` | `0` | No shortcut; every query goes straight to the unified index |

The mapping above mirrors `RETRIEVAL_MODES` in the client
([HeiChatClient.jsx:78-82](https://github.com/cgoul-code/chatbot-client-HEI20-v2/blob/main/client/src/HeiChatClient.jsx#L78-L82)) — see the
client README for example queries that exercise each mode.

### Legacy mode

The original setup. Only the article index is queried for grounding. The
`hvaerinnafor_qa_bank` index is still loaded but is used purely for the related-
questions suggestion list and for the fast follow-up path
(`agent: hvaerinnafor_related_qa`).

### Hybrid mode

Same Steg-1 cascade as legacy ([agent_workflow_answer.py:874-892](agent_workflow_answer.py#L874-L892)),
but the fallback uses the **unified** index instead of the article-only index:

```
question
   │
   ▼
Steg-1: psa_ssa_topics retriever
   │
   ├─ top score ≥ psa_ssa_threshold (0.65)
   │     └─► use those Q&A nodes as context, skip the main index
   │
   └─ otherwise
         └─► Steg-2: hvaerinnafor_unified — articles + Q&A nodes compete
```

The threshold is tunable per request via `psa_ssa_threshold` in the request body
(`<= 0` disables the cascade entirely — that's how Unified mode is configured).
When the cascade fires, the LLM sees the matched ung.no veileder answers as its
only context.

### Unified mode

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
embeddings rank highest. The Steg-1 `psa_ssa_topics` cascade is **disabled** in
Unified mode (the client sends `psa_ssa_threshold: 0`), so unified is the
cleanest A/B signal for what the unified index itself contributes — no
short-circuit can mask the comparison.

### Choosing a mode

| If you want… | Use |
|---|---|
| A/B baseline — article-only fallback behind the FAQ shortcut | Legacy |
| FAQ shortcut on, but articles + Q&A compete on misses | Hybrid |
| No shortcut — pure unified retrieval, articles and Q&A ranked together | Unified |

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

Streams an answer to a user question. The shape below matches what the HEI 2.0
client actually sends from
[HeiChatClient.jsx:941-965](https://github.com/cgoul-code/chatbot-client-HEI20-v2/blob/main/client/src/HeiChatClient.jsx#L941-L965)
— shared fields are identical across modes; only `vectorIndex` and
`psa_ssa_threshold` change.

**Headers:**
```
Content-Type: application/json
Cache-Control: no-cache
Accept: text/event-stream
```

**Shared fields (same for every request):**

| Field | Type | Source / typical value |
|---|---|---|
| `messages` | array | `[{ role: 'user', content: '<spørsmål>' }]` — current user input |
| `session_id` | string | uuid v4 generated per page load (`chatId`) |
| `clientType` | string | `"HEI"` |
| `agent` | string | `"hvaerinnafor"`, or `"hvaerinnafor_related_qa"` for related-question follow-ups |
| `similarity_top_k` | string | `"5"` |
| `similarity_cutoff` | string | `"0.75"` |
| `stream` | boolean | `true` |
| `from_related_q` | boolean | `true` if the user clicked a related-question suggestion, else `false` |
| `from_node_id` | string \| null | source node id when `from_related_q` is `true`, else `null` |
| **`vectorIndex`** | string | **per [retrieval mode](#retrieval-modes)** |
| **`psa_ssa_threshold`** | number | **per [retrieval mode](#retrieval-modes)** |
| `qa_bank_index` | string \| null | *Optional.* Explicit QA-bank index name (used for related-question suggestions and the fast follow-up path). If omitted or the named index isn't loaded, the server tries `{vectorIndex}_qa_bank` then falls back to `hvaerinnafor_qa_bank`. Set this to suppress the "QA-bank … not found" warning when running with a `vectorIndex` that has no matching `_qa_bank` (e.g. `hvaerinnafor_unified` → set `qa_bank_index: "hvaerinnafor_qa_bank"`). |
| `response_style` | string \| null | *Optional.* Override for the answer rewrite style. One of `"factual"` (skip rewrite), `"warm"`, `"supportive"`, `"crisis"`. Omit or use unknown value to auto-route from severity + stance. See [Response styles](#response-styles). |

**Per-mode values:**

| Mode | `vectorIndex` | `psa_ssa_threshold` |
|---|---|---|
| Legacy (`FAQ->Art`) | `"hvaerinnafor"` | `0.65` |
| Hybrid (`FAQ->Hyb`) | `"hvaerinnafor_unified"` | `0.65` |
| Unified (`Hyb`) | `"hvaerinnafor_unified"` | `0` |

**Example payload — Unified mode:**
```json
{
  "messages": [{ "role": "user", "content": "Kan jeg slutte på videregående?" }],
  "session_id": "f7c1e9...",
  "clientType": "HEI",
  "agent": "hvaerinnafor",
  "similarity_top_k": "5",
  "similarity_cutoff": "0.75",
  "vectorIndex": "hvaerinnafor_unified",
  "psa_ssa_threshold": 0,
  "qa_bank_index": "hvaerinnafor_qa_bank",
  "stream": true,
  "from_related_q": false,
  "from_node_id": null
}
```

Swap `vectorIndex` and `psa_ssa_threshold` to the values from the per-mode
table above to reproduce Legacy or Hybrid behavior. The client README has full
worked examples per mode.

**SSE events emitted:**

| Event name | Content |
|---|---|
| `answer` | Answer text delta (markdown) |
| `short_answer` | One-sentence summary |
| `references` | `[Title](url) \|\|IMG\|\| icon_url` per source |
| `related queries` | JSON array of follow-up question suggestions |
| `Refined query` | Rewritten version of the user's question |
| `query_status` | JSON snapshot of the pipeline's classification + retrieval state — see [query_status payload](#query_status-payload) below |
| `info` | Internal pipeline step names (for debugging) |
| `systeminfo` | Citation verification details (for debugging) |
| `Token usage` | Token counts and estimated NOK cost |
| `done` | Stream completed |
| `error` | Error message if something failed |

To use the **related-question fast path**, set `agent` to
`hvaerinnafor_related_qa` and include `from_node_id` (the node ID from a
previously emitted `related queries` item).

#### `query_status` payload

Emitted once per turn, near the end of the workflow inside
`emit_query_answer_references` ([agent_workflow_answer.py](agent_workflow_answer.py)).
Fires for **all** terminal paths — including the harm-routing branches
(`refuse_harm_to_others`, `help_after_harm`), since both join back into
`emit_query_answer_references` before END. This makes it the single,
reliable signal for what classification the turn ended up with.

The payload is a single JSON object delivered as `text` on the event:

```json
{
  "refined_query": "Hvordan virker p-piller?",
  "query_severity": "Green",
  "stance": "info_seeker",
  "harm_to_others_tense": "na",
  "response_style": "factual",
  "response_style_source": "auto",
  "relevancy_band": "Strong",
  "best_node_score": 0.812,
  "response_set": true,
  "validate_response_result": "Accepted"
}
```

| Field | Source | Notes |
|---|---|---|
| `refined_query` | `analyze_query` | Rewritten search-friendly form of the user's question |
| `query_severity` | `analyze_query` | `"Green"` / `"Yellow"` / `"Red"` / `""` |
| `stance` | `analyze_query` | `"info_seeker"` / `"affected_party"` / `"harm_to_others"` / `"ambiguous"` |
| `harm_to_others_tense` | `analyze_query` | `"planning"` / `"completed"` / `"unclear"` / `"na"`. `"na"` whenever stance ≠ `harm_to_others` |
| `response_style` | `apply_response_style` | Final effective style — `"factual"` / `"warm"` / `"supportive"` / `"crisis"`. Empty string `""` when the rewrite step was bypassed (harm routes, Rejected turns, very short answers) |
| `response_style_source` | `apply_response_style` | How the style was picked: `"auto"` (server auto-routed), `"override"` (client sent a valid `response_style`), `"forced_red"` (severity=Red safety floor clamped to `crisis`), or `""` (rewrite step skipped) |
| `relevancy_band` | derived in `emit_query_answer_references` | `"Strong"` / `"Medium"` / `"Rejected"` — computed from the best retrieval score across all completed subqueries (`_classify_relevancy`). Empty `""` when no retrieval ran (e.g. harm routes) |
| `best_node_score` | derived in `emit_query_answer_references` | `max(response_validity_index)` over the completed subqueries. `0.0` when no retrieval ran |
| `response_set` | state | Boolean — `true` iff the LlamaIndex `Response` object on state is non-null. Useful as a "retrieval actually happened" signal |
| `validate_response_result` | `synthesizer` / `fast_single` / harm nodes | `"Accepted"` when a grounded (or harm-routed) answer was produced, `"Rejected"` when the answer is a `CANNOT_ANSWER_PLACEHOLDER` |

Typical use on the client: surface these fields behind a debug info-icon
on each turn, classify test fixtures by stance/tense, or filter
production logs for `validate_response_result == "Rejected"`. The HEI
2.0 frontend wires this into a per-message popover; see
[HeiChatClient.jsx](https://github.com/cgoul-code/chatbot-client-HEI20-v2/blob/main/client/src/HeiChatClient.jsx)
for the `case 'query_status':` handler.

Per-bucket behavior on harm routes:

- `stance` will be `"harm_to_others"`, `harm_to_others_tense` set to `"planning"` / `"completed"` / `"unclear"`.
- `validate_response_result` is `"Accepted"` — the safety nodes produce their own grounded-style text, they don't go through claim verification.
- `response_style` / `response_style_source` are both `""` since `apply_response_style` is bypassed for harm routes.
- `relevancy_band` is `""` and `best_node_score` is `0.0` — no retrieval was performed.

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

## Safety routing (harm_to_others)

`analyze_query` doesn't only assess severity — it also classifies the user's
**stance** in the situation. This is a separate axis from severity and is what
decides whether the question should be answered from the knowledge base at all,
or routed to a dedicated safety node.

### Stance

| Stance | Meaning | Example |
|---|---|---|
| `info_seeker` | Asks for general knowledge or advice, no specific actor | "Hva sier loven om deling av nakenbilder?" |
| `affected_party` | Describes something that affects the user themselves (victim, pressured, worried) | "Eksen min truer med å dele nakenbilder av meg" |
| `harm_to_others` | Asks for help performing an illegal/harmful act on another person | "Jeg vil dele nakenbilder av eksen min" |
| `ambiguous` | Unclear who is the actor and who is the target | — |

The classifier is told to break ties toward `harm_to_others` when a concrete
intention is expressed ("jeg vil …", "hva er enkleste måten å …"), and toward
`info_seeker` for pure knowledge questions ("hva sier loven om …").

### Tense (only when stance = `harm_to_others`)

When stance is `harm_to_others`, a second classification picks the **tense** of
the harmful action. This decides which safety node handles the response:

| Tense | Triggers on | Routes to |
|---|---|---|
| `planning` | Future/modal verbs: "vil", "skal", "hvordan kan jeg", "hva er enkleste måten å …" | `refuse_harm_to_others` |
| `completed` | Past tense / present perfect: "har sendt", "delte", "filmet", "installerte" | `help_after_harm` |
| `unclear` | Ambiguous (e.g. "har bildene og vurderer å dele dem") | `refuse_harm_to_others` |
| `na` | Stance is not `harm_to_others` | (normal flow) |

### Safety nodes

Both nodes are **LLM-driven** with strict system prompts in
[registry.py](registry.py) (`REFUSE_HARM_PROMPT`, `HELP_AFTER_HARM_PROMPT`),
and fall back to hardcoded messages (`HARM_REFUSAL_ANSWER`,
`HELP_AFTER_HARM_ANSWER`) if the LLM call fails. Both nodes deliberately
**skip retrieval and empathy_rewrite**:

- Retrieval is skipped because the knowledge base is written for victims and
  general info, not for perpetrators — answering a perpetrator question from
  the victim corpus produces the wrong frame.
- `empathy_rewrite` is skipped because its prompt contains a victim-oriented
  example for the "deling av nakenbilder uten samtykke" topic which would pull
  the response back into the victim frame regardless of stance.

**`refuse_harm_to_others`** — for planning/unclear cases. Acknowledges the
underlying feeling briefly, states clearly that the act is illegal or harmful
(LLM cites Norwegian penal code when it can be precise: §267a image sharing,
§282 intimate partner abuse, §263/266 threats, §297–299 sexual assault,
§305 grooming, §312 incest, §145 stalkerware, etc.), **never describes how
the act could be performed**, and points to 2–3 concrete resources (Mental
Helse Ungdom 116 123, Kors på halsen 16 111, Slettmeg.no, politiet, Sex og
samfunn, a trusted adult).

**`help_after_harm`** — for completed cases. **Not** a refusal — the harm is
already done, and a cold refusal helps no one. Acknowledges that reaching out
takes courage, gives a realistic picture of the legal/personal consequences,
and provides concrete next steps focused on damage limitation: stop further
spread (Slettmeg.no), talk to a trusted adult, consider proactively contacting
police or a lawyer, think about how to limit harm for the affected party, and
get help for the user themselves.

---

## Response styles

The `apply_response_style` node (formerly `empathy_rewrite`) rewrites the
final answer to fit the situation. Instead of one prompt with severity-
conditional hints, there are now **four named profiles**, each with its own
prompt and its own examples — so no profile leaks the wrong tone into another.

| Style | When used | Characteristics |
|---|---|---|
| `factual` | `info_seeker` + Green (or unmatched) | **Rewrite is skipped entirely.** Saves a full LLM call. The grounded answer is returned as-is. |
| `warm` | `info_seeker` + Yellow, or `affected_party` + Green | Light normalization, max one acknowledging sentence, then facts. Conversational but not heavy. |
| `supportive` | `affected_party` + Yellow | Opens with validation ("det du beskriver høres vanskelig ut"), then facts, closes with presence ("du står ikke alene"). |
| `crisis` | Severity = Red, any stance | Direct support first, single clearest next action, short and focused. Avoids long bullet menus. |

### Auto-routing

`apply_response_style` calls `pick_response_style(severity, stance)`
([agent_workflow_answer.py](agent_workflow_answer.py)) to pick a profile
deterministically. The full matrix:

| stance \ severity | Green (or empty) | Yellow | Red |
|---|---|---|---|
| `info_seeker` | `factual` (skip) | `warm` | `crisis` |
| `affected_party` | `warm` | `supportive` | `crisis` |
| `ambiguous` | `factual` (skip) | `warm` | `crisis` |
| `harm_to_others` | *(never reaches this node — routed to `refuse_harm_to_others` or `help_after_harm` instead)* | *(same)* | *(same)* |

Two asymmetries worth noting:

- **`Red` always routes to `crisis`**, regardless of stance. Severity is
  checked before stance in `pick_response_style`, so the stance row doesn't
  matter on Red.
- **`affected_party + Green` is `warm`, not `factual`.** A self-affected
  question gets at least a light empathy pass even if the topic is
  classified as preventive/Green — we want to acknowledge the user's
  situation, not just dispense facts.

This routing is policy, not an LLM decision. It's predictable and auditable.

### Client override

The client can override the auto-routing by sending `response_style` in the
`POST /chat` request body, with one of: `"factual"`, `"warm"`,
`"supportive"`, `"crisis"`. Omit the field (or send an unknown value) to use
auto-routing.

**Precedence:**
1. **Safety floor:** if `severity == "Red"`, the style is always clamped to
   `crisis` — *even if the client requested something else*. A client
   override of e.g. `factual` on an acute crisis would produce a cold,
   fact-only response on a Red question, which is never appropriate. The
   floor only matters when an override is present (auto-routing returns
   `crisis` on Red anyway).
2. Valid `response_style` in request body → that style wins (subject to
   the safety floor above).
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

**Client UI (HEI 2.0):** the chatbot client exposes a pill row mirroring
this — `Auto / Fakta / Vennlig / Støtte / Krise`. "Auto" sends nothing
(field omitted via spread, so the server auto-routes). The other pills map
1:1 to the four style values. Mode is read per-request; already-rendered
turns keep whatever style they were sent under. See
[`RESPONSE_STYLES` in HeiChatClient.jsx](https://github.com/cgoul-code/chatbot-client-HEI20-v2/blob/main/client/src/HeiChatClient.jsx).

**Debugging:** `apply_response_style` emits an `info` SSE event indicating
the chosen path:
- `Response style: <style> (client override)` — explicit value won
- `Response style: <style> (auto)` — fell through to `pick_response_style`
- `Response style: forced crisis (was <style>, severity=Red)` — safety
  floor activated; the client (or auto-routing) picked a non-crisis style
  on a Red question, and it was clamped

This makes it easy to see in the stream whether a surprising tone came from
auto-routing, an unexpected client value, or the safety floor.

**Unknown values are safe:** if the client sends a typo or a style we've
removed (`response_style: "casual"`), the server doesn't crash — the value
fails the `in RESPONSE_STYLES` check and auto-routing takes over.

### Why the rewrite step exists at all

The grounded answer from `query_grounded` / `synthesizer` is fact-correct but
often tonally neutral. Rewriting it lets us vary tone independently from
retrieval — same facts, different presentation. The `factual` skip path
means low-emotional questions don't pay the rewrite cost (~1 extra LLM
call, ~100-300 ms latency).

### Skipped paths

- The two harm safety nodes (`refuse_harm_to_others`, `help_after_harm`)
  bypass `apply_response_style` entirely — they produce already-styled
  text via their own LLM prompts. Running a warm/supportive rewrite on
  top of a refusal would soften it inappropriately.
- The `factual` style is a no-op pass-through — no LLM call, just emits
  `response_style: "factual"` so downstream debug tools can see what
  happened.

---

## Adding new content

1. **New articles** — add source documents to the `hvaerinnafor` index using
   the existing ingestion pipeline. Each document needs `title`, `url`, and
   optionally `icon_url` in its metadata. For unified mode, the same articles
   must also be re-indexed into `hvaerinnafor_unified` so the synthetic-question
   aliases and cross-pollination step run on them.

2. **New Q&A pairs** — add entries to the `hvaerinnafor_qa_bank` index. Each
   node needs `answer`, `short_answer`, `severity`, `category`, `valid: 1`,
   and `from_doc_id` pointing to the source article.

3. **New ung.no Q&A entries** — add to `psa_ssa_topics` (used by the hybrid
   tier-1 cascade) and re-run the unified-index build so the entries are
   ingested as Q&A nodes and considered for cross-pollination onto article
   chunks.

4. **New categories** — add to the `categories` list in `answer_utils.py`.

5. **New topics** — update the prompt and severity examples in
   `ANALYZE_QUERY_PROMPT` ([registry.py](registry.py)) and the category
   list passed in from `answer_utils.py`.

---

## Deployment

The app is deployed as an **Azure App Service** (Linux, Python 3.11). On
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