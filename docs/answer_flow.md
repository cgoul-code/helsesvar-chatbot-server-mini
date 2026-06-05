# Svarflyt — alle spor et spørsmål kan følge

Diagrammet under viser hele `answer_workflow` (definert i
[agent_workflow_answer.py](../agent_workflow_answer.py)): fra et spørsmål kommer
inn, gjennom stance-klassifiseringen, og ut til ferdig svar.

## To typer skille

- 🔴 **Innholds-/rute-skille** — `harm_to_others` (begge tempus) og
  `expresses_prejudice` får hver sin LLM-node fordi de trenger *annet innhold*.
  De forlater hovedflyten tidlig og hopper over både RAG og tone-omskriving.
- 🔵 **Samme RAG-spor** — `info_seeker` og `affected_party` går *begge* hit.
  Stance avgjør ikke veien; bare `needs_subqueries` gjør det
  (`fast_single` vs `orchestrator`).
- 🟢 **Tone-skille** — først i `apply_response_style` skiller `info_seeker` seg
  fra `affected_party`, og kun i valg av stil (supportive / warm / factual / crisis).

## Diagram

```mermaid
flowchart TD
    START([Spørsmål]) --> AQ[analyze_query<br/>stance · severity · tense · needs_subqueries]

    subgraph CONTENT["🔴 INNHOLDS-/RUTE-SKILLE — egne LLM-spor, hopper over RAG"]
        HAH[help_after_harm]
        RHO[refuse_harm_to_others]
        AP[address_prejudice]
    end

    AQ -->|harm_to_others · completed| HAH
    AQ -->|harm_to_others · ellers| RHO
    AQ -->|expresses_prejudice| AP

    subgraph RAG["🔵 SAMME RAG-SPOR — info_seeker & affected_party deler dette"]
        ORC[orchestrator] --> QG[query_grounded ×N<br/>RAG + entailment] --> SYN[synthesizer]
        FS[fast_single]
    end

    AQ -->|"needs_subqueries = true"| ORC
    AQ -->|"ellers (enkelt)"| FS

    SYN --> ARS
    FS --> ARS

    subgraph TONE["🟢 TONE-SKILLE — her, og BARE her, skiller info_seeker seg fra affected_party"]
        ARS{{"apply_response_style<br/>pick_response_style(severity, stance)"}}
        ARS -->|affected_party + Yellow| ST1[supportive]
        ARS -->|"affected_party+Green / info_seeker+Yellow"| ST2[warm]
        ARS -->|info_seeker + Green| ST3[factual · ingen omskriving]
        ARS -->|severity = Red| ST4[crisis · safety floor]
    end

    ST1 --> EMIT[emit_query_answer_references]
    ST2 --> EMIT
    ST3 --> EMIT
    ST4 --> EMIT
    HAH --> EMIT
    RHO --> EMIT
    AP --> EMIT

    EMIT --> RQ[related_queries_dialog] --> END([Ferdig])

    classDef llm fill:#fde2e2,stroke:#c0392b,color:#000;
    classDef rag fill:#e2f0fd,stroke:#2980b9,color:#000;
    classDef style fill:#e8f8e8,stroke:#27ae60,color:#000;
    class HAH,RHO,AP llm;
    class FS,ORC,QG,SYN rag;
    class ST1,ST2,ST3,ST4 style;
```

## Stance → spor

| stance | spor | hvordan svaret lages |
|--------|------|----------------------|
| `harm_to_others` (completed) | **help_after_harm** | LLM, skadebegrensning + hjelpetjenester. Ingen RAG |
| `harm_to_others` (planning/unclear) | **refuse_harm_to_others** | LLM, avvisning + lovverk. Ingen RAG |
| `expresses_prejudice` | **address_prejudice** | LLM, møter holdning uten å validere den. Ingen RAG |
| `info_seeker` / `affected_party` (enkelt) | **fast_single** | RAG, ett oppslag |
| `info_seeker` / `affected_party` (sammensatt) | **orchestrator → query_grounded → synthesizer** | RAG, deles i delspørsmål, hver med sitatsjekk + entailment-gate, så flettes |

## Tone-valg (`pick_response_style`)

| stance | severity | → stil |
|--------|----------|--------|
| `affected_party` | Yellow | supportive |
| `affected_party` | Green | warm |
| `info_seeker` | Yellow | warm |
| `info_seeker` / `ambiguous` | Green | factual (ingen omskriving) |
| *hva som helst* | **Red** | crisis (safety floor — overstyrer alt) |
