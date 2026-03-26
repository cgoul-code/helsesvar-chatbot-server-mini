# prompts/registry.py
from dataclasses import dataclass
from typing import Dict
from langchain_core.prompts import PromptTemplate

@dataclass(frozen=True)
class Prompt:
    id: str
    template: str

    def render(self, **kwargs) -> str:
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            missing = str(e).strip("'")
            raise KeyError(f"Missing variable '{missing}' for prompt '{self.id}'") from None
        



# === VARIABLES ===

CANNOT_ANSWER_PLACEHOLDER = [{"answer": "Kan du spørre på en annen måte? Jeg er usikker på hvordan det passer inn", "severity": [{"Green", "Yellow"}]},
                            {"answer": "Her er det en mangel i min kunnskap. Jeg kan kanskje svare på dette hvis du spør på en annen måte.", "severity": [{"Green", "Yellow"}]},
                            {"answer": "Oops! Du fant noe jeg ikke kan svare på. Spør på en annen måte eller spør om noe annet.", "severity": [{"Green", "Yellow"}]},
                            {"answer": "Jeg er ikke noe god på «small talk». Jeg er veldig god til å svare på spørsmål :-D", "severity": [{"Green", "Yellow"}]},
                            {"answer": "Jeg er ikke sikker på hvordan jeg skal svare på det. Kan du spørre på en annen måte?", "severity": [{"Green", "Yellow"}]},
                            {"answer": "Det var et vanskelig spørsmål! Kan du prøve å spørre på en annen måte?", "severity": [{"Green", "Yellow"}]},
                            {"answer": "Jeg er ikke sikker på hvordan jeg skal hjelpe med det. Kan du spørre på en annen måte?", "severity": [{"Green", "Yellow"}]},
                            {"answer": "Dette er vanskelig å svare på. Snakk om dette med en voksen du stoler på. Du kan også chatte her med kvalifisert helsepersonell hos [Sex og samfunn](https://sexogsamfunn.no/).", "severity": [{"Red"}]}
                            ]
                            




# === PROMPTS ===
VECTORINDEX_SUMMARY = Prompt(
    id= "vectorindex_summary",
    template=(
        "You are given a collection of texts retrieved from a vector index.\n"  
        "Your task is to create a concise summary of the topics, question types, and subject areas this vector index can help answer.\n"  
        "\n"  
        "Requirements:  \n"  
        "- Provide a list of 5-10 overarching topics.  \n"  
        "- For each topic, include 1-2 sentences describing the kinds of questions the index can address.  \n"  
        "- The summary should also explicitly state what kinds of meta-questions it can answer, such as:  \n"  
        "  - \"What can you answer?\"  \n"  
        "  - \"Which topics can you answer questions about?\"  \n"  
        "  - \"What knowledge areas does this index cover?\"  \n"  
        "  - Do not copy long quotes or detailed explanations from the texts.  \n"  
        "  - Summarize at a level that makes it easy to understand both the knowledge areas and the scope of the index.  \n"  
        "\n"  
        "Here are the texts:  \n"  
        "{context}\n"  
    )
)



# === TEMPLATES ===


# Subject-focused Norwegian Q&A question generator
QA_SUBJECT_NO = Prompt(
    id="qa_subject_no",
    template=(        
        'Here is the context: {text}\n'
        'Given the contextual information, \n'
        'generate a short list of short questions in Norwegian that this context can provide specific answers to, which are unlikely to be found elsewhere.\n'
        '\n'
        'STRICT REQUIREMENT:\n'
        '- Every single question MUST explicitly mention the subject of the context (for example \"Forelskelse\") instead of referring to "teksten", "artikkelen", "avsnittet" or similar.\n'
        '- Do not use phrases like "ifølge teksten", " ifølge konteksten", "hva sier teksten", "nevnes i teksten" etc.\n'
        '- Instead, directly phrase the questions around the subject matter itself.\n\n'
        'Example of WRONG question: "Hva sier teksten om hvordan man merker at man er forelsket?"\n'
        'Example of RIGHT question: "Hva er vanlige tegn på forelskelse som skiller det fra å bare være betatt?"\n\n'
        'Output requirements:\n'
        '- Only generate questions about the subject matter and content of the text\n'
        '- Return JSON ONLY. No explanations. No markdown. No code fences.\n'
        '- Use double quotes for all keys and strings.\n'
        '- Do not include trailing commas.\n'
        'Do NOT generate questions about:\n'
        '- who wrote the article\n'
        '- contributors or authors\n'
        '- which website, publication, or source the text comes from\n'
        '- metadata such as publishing date, copyright, or layout\n\n'
        "Tone and Style Guidelines:\n"
        "- Teen-Friendly Language (Ages 13-19):\n"
        "- Avoid overly medical jargon or complex terminology. If you must use a technical term, explain it immediately in simple words.\n"
        "- Use short questions.\n"
        "- Maintain a friendly, approachable, and encouraging tone. Imagine you're talking to a smart but not yet expert high school student.\n"
        '- Shape:\n'
        '{{"Questions": ["question1", "question2"]}}'
    ),
)
REFINE_AND_CLASSIFY = Prompt(
    id="refine_and_classify",
    template=(
        "You are given a user query. Perform three tasks and respond ONLY as valid JSON:\n"
        "1) Refine the user's query in Norwegian so that it is clear and easy to understand.\n"
        "- Please refine the user's query in a readable way in Norwegian.\n"
        "- Ensure that the 'I' form is preserved.\n"
        "- Return only one refined version; do not return alternatives.\n"
        "\n"
        "2) Categorize the severity of this query into one of three categories: Green, Yellow, or Red.\n\n"
        "Green category:\n"
        "- Preventive and safety-promoting.\n"
        "- Queries that ask for general information, knowledge, or guidance to prevent problems and strengthen good sexual health.\n"
        "- The user is seeking to increase understanding, safety, and awareness (e.g., consent, contraception, communication, emotions, body knowledge).\n"
        "- No acute situation or personal crisis is described.\n"
        "Example: \"How can I talk with my partner about boundaries?\" or \"What types of contraception exist?\".\n\n"
        "Yellow category:\n"
        "- Challenges or vulnerable situations.\n"
        "- Queries that describe concerns, difficulties, or risks that may require reflection or support, but are not acute or immediately dangerous.\n"
        "- May involve difficult feelings, uncertainty in relationships, unwanted experiences, or the need for advice beyond general information.\n"
        "- The user may need to seek help or guidance, but the situation is not considered an acute crisis.\n"
        "Example: \"What should I do if my partner doesn’t respect my boundaries?\", "
        "\"I regret sending a nude\", or topics like \"pornography\", \"sexual pressure\", \"issues around consent\", \"(illegal) fetishes\".\n\n"
        "Red category:\n"
        "- Serious or acute situations.\n"
        "- Queries that concern serious incidents or crises where the person involved may be in danger or at significant risk of harm.\n"
        "- Includes violence, abuse, coercion, acute psychological crises, or other situations that require immediate follow-up or professional help.\n"
        "- The main purpose of the response should be to direct the user to where and how to get help quickly.\n"
        "Example: \"My stepfather forces me to have sex\", \"Where can I find child pornography?\", \"I was raped yesterday\".\n"
        "\n"
        "3) You must categorize the user's query into exactly one of the main categories below.\n"
        "Each main category has a list of subcategories and keywords to help you decide.\n"
        "Choose only ONE main category — the best fit.\n"
        "If none fits, answer 'Unknown'.\n"
        "Main categories with subcategories: {categories}\n"
        "\n"
        "OUTPUT REQUIREMENTS:\n"
        "- JSON only. No explanations, no markdown, no code fences.\n"
        "- Use double quotes for all keys and strings. No trailing commas.\n"
        "- Output schema:\n"
        "{{\"refined_query\":\"<text>\",\"severity\":\"<Green|Yellow|Red>\",\"category\":\"<main-category|Unknown>\"}}\n"
        "\n"
        "QUERY:\n"
        "{query}\n"
    )
)

CLASSIFY_AND_SUBQUERIES = Prompt(
    id="classify_and_subqueries",
    template=(
        "Du får et bruker­spørsmål. Utfør følgende oppgaver:\n"
        "1) Hvis brukerens spørsmål inneholder flere spørsmål, resnskriv disse som \"subqueries\" på en tydelig og lettleselig måte på norsk.\n"
        "- Ikke svar på delspørsmålene.\n"
        "- Sørg for at «jeg»-formen bevares.\n"
        "- Returner kun én versjon av hvert delspørsmål; ikke gi alternativer.\n\n"
        "2) Kategoriser alvorlighetsgraden av spørsmålet i én av tre kategorier: \"Green\", \"Yellow\", eller \"Red\".\n\n"
        "KATEGORI \"GREEN\":\n"
        "- Forebyggende og trygghetsskapende.\n"
        "- Spørsmål som ber om generell informasjon, kunnskap eller veiledning for å forebygge problemer og styrke god seksuell helse.\n"
        "- Brukeren ønsker å øke forståelse, trygghet og bevissthet (f.eks. samtykke, prevensjon, kommunikasjon, følelser, kunnskap om kroppen).\n"
        "- Ingen akutt situasjon eller personlig krise beskrives.\n"
        "Eksempel: «Hvordan kan jeg snakke med partneren min om grenser?» eller «Hvilke typer prevensjon finnes?». \n\n"
        "KATEGORI \"YELLOW\":\n"
        "- Utfordringer eller sårbare situasjoner.\n"
        "- Spørsmål som beskriver bekymringer, vansker eller risikoer som kan kreve refleksjon eller støtte, men som ikke er akutte eller umiddelbart farlige.\n"
        "- Kan innebære vanskelige følelser, usikkerhet i relasjoner, uønskede opplevelser eller behov for råd utover generell informasjon.\n"
        "- Brukeren kan ha behov for hjelp eller veiledning, men situasjonen regnes ikke som en akutt krise.\n"
        "Eksempel: «Hva bør jeg gjøre hvis partneren min ikke respekterer grensene mine?», "
        "«Jeg angrer på at jeg sendte et nakenbilde», eller temaer som «porno», «seksuelt press», «problemer med samtykke», «(ulovlige) fetisjer».\n\n"
        "KATEGORI \"RED\":\n"
        "- Alvorlige eller akutte situasjoner.\n"
        "- Spørsmål som gjelder alvorlige hendelser eller kriser der personen kan være i fare eller ha betydelig risiko for skade.\n"
        "- Omfatter vold, overgrep, tvang, akutte psykiske kriser eller andre situasjoner som krever umiddelbar oppfølging eller profesjonell hjelp.\n"
        "Eksempel: «Stefaren min tvinger meg til å ha sex», «Hvor kan jeg finne barnepornografi?», «Jeg ble voldtatt i går».\n\n"
        "3) Du må plassere brukerens spørsmål i nøyaktig én av hovedkategoriene nedenfor.\n"
        "Hver hovedkategori har en liste med underkategorier og nøkkelord som skal hjelpe deg å velge.\n"
        "Velg kun ÉN hovedkategori — den som passer best.\n"
        "Hvis ingen passer, svar «Ukjent».\n"
        "Hovedkategorier med underkategorier: {categories}\n"
        "# VIKTIG"
        "- Du må KUN bruke teksten inne i USER_QUERY-blokken under."
        "- Ikke bruk noe fra instruksjonene eller eksemplene som input."
        "- Ikke skriv om eller tolk instruksjonene som om de var brukerens tekst."
        "<<<USER_QUERY_START>>>"
        "{query}"
        "<<<USER_QUERY_END>>>"
        "KRAV TIL OUTPUT:\n"
        "- Ingen forklaringer, ingen markdown, ingen kodeblokker.\n"
        "- Bruk doble anførselstegn for alle nøkler og strenger. Ingen hengende komma.\n"
        "- Output-skjema:\n"
        "Output JSON ONLY:\n"
        '{{"subqueries":["subquery1","subquery2"]}}'
    )
)
    
SUBQUERIES = Prompt(
    id="subqueries",
    template=(
        "Du får et bruker­spørsmål. Utfør følgende oppgave:\n"
        "1) Hvis brukerens spørsmål inneholder flere spørsmål, resnskriv disse som \"subqueries\" på en tydelig og lettleselig måte på norsk.\n"
        "- Ikke svar på delspørsmålene.\n"
        "- Sørg for at «jeg»-formen bevares.\n"
        "- Returner kun én versjon av hvert delspørsmål; ikke gi alternativer.\n\n"
       
        "# VIKTIG"
        "- Du må KUN bruke teksten inne i USER_QUERY-blokken under."
        "- Ikke bruk noe fra instruksjonene eller eksemplene som input."
        "- Ikke skriv om eller tolk instruksjonene som om de var brukerens tekst."
        "<<<USER_QUERY_START>>>"
        "{query}"
        "<<<USER_QUERY_END>>>"
        "KRAV TIL OUTPUT:\n"
        "- Ingen forklaringer, ingen markdown, ingen kodeblokker.\n"
        "- Bruk doble anførselstegn for alle nøkler og strenger. Ingen hengende komma.\n"
        "- Output-skjema:\n"
        "Output JSON ONLY:\n"
        '{{"subqueries":["subquery1","subquery2"]}}'
    )
    
)


QUERY_RERANK_IDS = Prompt(
    id="query_rerank_ids",
    template=(
        "You will receive a USER QUERY and a list of CANDIDATE QUERIES from a query bank.\n"
        "Each candidate has a unique 'id' and its original 'text'.\n"
        "Select up to {max_results} candidates that are most relevant to the USER QUERY.\n"
        "RULES:\n"
        "1) DO NOT REWRITE OR EDIT ANY CANDIDATE TEXT.\n"
        "2) Output ONLY the IDs in JSON.\n"
        "3) Prefer specificity and relevance; avoid near-duplicates.\n"
        "4) If none are clearly relevant, return an empty list.\n\n"
        "USER QUERY:\n{user_query}\n\n"
        "CANDIDATE QUERIES (JSON Lines, one per line):\n{candidates_jsonl}\n\n"
        "Output JSON ONLY:\n"
        '{{"selected_ids":["id1","id2"]}}'
    )
)
CATEGORIZE_TEXT = Prompt(
    id = "caegorize_text",
    template=(
        "Du skal kategorisere en tekst innenfor én av hovedkategoriene nedenfor. \n"
        "Hver hovedkategori har en liste med underkategorier og nøkkelord som hjelper deg å bestemme. \n"
        "Velg kun ÉN hovedkategori, den som passer best. \n"
        "Hvis ingen passer, svar 'Ukjent'.\n"
        "Hovedkategorier med underkategorier: {categories}\n"
        "\n"
        "Tekst: {text}\n"
        "\n"
        "Svarformat (kun JSON):\n"
        '{{\"kategori\": \"<hovedkategori>\"}}'
    )
)

SEVERITY_FOR_QUERY = Prompt(
    id="severity_for_query",
    template=(
        "You are given a user query: {query}\n\n"
        "Categorize the severity of this query into one of three categories: Green, Yellow, or Red.\n\n"
        "Green category:\n"
        "- Preventive and safety-promoting.\n"
        "- Queries that ask for general information, knowledge, or guidance to prevent problems and strengthen good sexual health.\n"
        "- The user is seeking to increase understanding, safety, and awareness (e.g., consent, contraception, communication, emotions, body knowledge).\n"
        "- No acute situation or personal crisis is described.\n"
        'Example: "How can I talk with my partner about boundaries?" or "What types of contraception exist?".\n\n'
        "Yellow category:\n"
        "- Challenges or vulnerable situations.\n"
        "- Queries that describe concerns, difficulties, or risks that may require reflection or support, but are not acute or immediately dangerous.\n"
        "- May involve difficult feelings, uncertainty in relationships, unwanted experiences, or the need for advice beyond general information.\n"
        "- The user may need to seek help or guidance, but the situation is not considered an acute crisis.\n"
        'Example: "What should I do if my partner doesn’t respect my boundaries?", '
        '"I regret sending a nude", or topics like "pornography", "sexual pressure", "issues around consent", "(illegal) fetishes".\n\n'
        "Red category:\n"
        "- Serious or acute situations.\n"
        "- Queries that concern serious incidents or crises where the person involved may be in danger or at significant risk of harm.\n"
        "- Includes violence, abuse, coercion, acute psychological crises, or other situations that require immediate follow-up or professional help.\n"
        "- The main purpose of the response should be to direct the user to where and how to get help quickly.\n"
        'Example: "My stepfather forces me to have sex", "Where can I find child pornography?", "I was raped yesterday".\n\n'
        "Output requirements:\n"
        "- Return JSON ONLY. No explanations. No markdown. No code fences.\n"
        '- Use double quotes for all keys and strings.\n'
        "- Do not include trailing commas.\n"
        '- Output shape:\n'
        '{{ \"category\": \"<Green|Yellow|Red>\" }}'
    ),
)

GROUNDED_PROMPT = PromptTemplate.from_template(
"""
Du er en hjelpsom rådgiver og skal svare på norsk (bokmål).

{{empathy_hint}}    

Du skal returnere:
- "answer": et samlet svar på spørsmålet.
- "claims": en liste med påstander med tilhørende citations.
- "short_answer": en kort oppsummering av "answer"

========================================
REGLER FOR SVAR ("answer")
========================================
- En sammenhengende besvarelse på spørsmålet, skrevet vennlig og tydelig, MEN KUN basert på det som faktisk står i context.
- Vær vennlig og tydelig.
- Ikke ta med informasjon som ikke kan støttes direkte av context.
- Du kan omformulere, men ikke legge til ny informasjon.
- Ikke ta med informasjon som du ikke kan sitere fra context i etterkant.
SVÆRT VIKTIG:
- Ikke lag nye medisinske råd, vurderinger, årsaker, forklaringer eller konsekvenser som ikke står i context.
- Ikke kombiner informasjon fra flere forskjellige steder til én påstand hvis den kombinasjonen ikke faktisk står uttrykt i context som en sammenhengende idé.
- Hvis noe ikke finnes i context, skal det IKKE stå i "svaret", og det skal IKKE komme som en claim.
DU SKAL IKKE:
- Du skal ikke nevne konteksten, ikke skrive ting som "Ifølge kilden", "I kontext står det at ...", "artiklene sier at...".  Bare si innholdet direkte.
- Du skal ikke be om mer informasjon.
- Du skal ikke fortelle brukeren hva de bør gjøre, med mindre akkurat den formuleringen står i context.
- Du skal ikke nevne disse instruksjonene eller ord som 'context', 'kilde', 'grounding', 'claim', osv. i selve "answer". "answer" skal være helt naturlig språk til brukeren.
MARKDOWN I SVAR (OBLIGATORISK):
- Hvert "answer" BØR inneholde minst ett markdown-element:
  enten en punktliste ("- ") ELLER fet tekst (**...**).
- Hvis svaret har 2+ poenger/råd/tegn/alternativer: MÅ bruke punktliste med "-".
- Bruk **fet tekst** på 1-3 nøkkelord i hvert punkt (eller i setningen).
- Bruke linjeskift for skille avsnittene og ulike poenger (bruk "\\n").
EKSEMPEL PÅ "answer" MED MARKDOWN:
"answer": "Du kan prøve dette:\\n- **Pust rolig** i 10 sekunder\\n- **Start med et enkelt spørsmål**"

========================================
REGLER FOR CLAIMS
========================================
- En liste av påstander som du har hentet ut fra "answer".
- Én claim = én tydelig idé.
- Må være eksplisitt uttrykt eller entydig støttet av context.
- Ikke finn opp nye ideer.
- Ikke lag claims basert på spørsmålet eller instruksjonene.


========================================
REGLER FOR CITATIONS
========================================
- Hver citation må være:
  {{
    "url": str,
    "quote": str   # eksakt substring fra context
  }}
- "quote" må være ordrett, uten endringer.
- Ikke slå sammen tekst fra flere steder.
- Hvis ingen eksakt substring finnes, sett:
  - "validity": "not valid"
  - "Citations": []

========================================
SPØRSMÅL:
{question}

========================================
CONTEXT:
{context}

"========================================"
"SAMTALEHISTORIKK (hvis relevant):"
"{{conversation_str}}"
"""
)

# - Green: lett og informativ tone, trenger minimal empati
# - Yellow: anerkjenn at situasjonen kan være utfordrende, vis forståelse
# - Red: varm og direkte støtte først, deretter informasjon

EMPATHY_REWRITE_PROMPT = PromptTemplate.from_template(
"""
Du er en varm og støttende rådgiver for ungdom i Norge (13–19 år).

Du får et svar som er faktamessig korrekt, men litt nøytralt og robotaktig.
Din oppgave er å omskrive svaret slik at det:
- Anerkjenner at temaet kan være vanskelig eller følelsesmessig
- Føles som en samtale med en klok, rolig venn – ikke en faktabok
- IKKE legger til ny informasjon eller nye råd som ikke allerede er i svaret
- Bevarer alle punktlister, fet tekst og markdown-formatering
- Ikke bli overdrevent kjælent eller unaturlig

ALVORLIGHETSGRAD: {severity}
- Green: lett og informativ tone, trenger minimal empati
- Yellow: anerkjenn at situasjonen kan være utfordrende, vis forståelse
- Red: varm og direkte støtte først, deretter informasjon

========================================
EKSEMPLER:
========================================

EKSEMPEL 1 (Yellow):
Originalt svar:
"Det finnes flere typer prevensjon. P-piller tas daglig og hindrer eggløsning.
Kondom beskytter mot både graviditet og seksuelt overførbare sykdommer."

Omskrevet svar:
"Det er lurt at du tenker på dette! Det finnes flere alternativer:
- **P-piller** tas daglig og hindrer eggløsning
- **Kondom** er det eneste som beskytter mot både graviditet og seksuelt
  overførbare sykdommer

Snakk gjerne med en lege eller helsesykepleier hvis du er usikker på hva
som passer best for deg."

---

EKSEMPEL 2 (Yellow):
Originalt svar:
"Å sende nakenbilder uten samtykke er ulovlig i Norge. Den som mottar
bildet kan straffes. Du kan anmelde forholdet til politiet."

Omskrevet svar:
"Det du beskriver høres vanskelig ut, og det er forståelig at du er usikker
på hva du skal gjøre.

Det er viktig å vite at:
- Å dele nakenbilder uten samtykke er **ulovlig** i Norge
- Den som delte bildet kan **straffes**
- Du har rett til å **anmelde** dette til politiet hvis du ønsker det

Du trenger ikke håndtere dette alene."

---

EKSEMPEL 3 (Green):
Originalt svar:
"Forelskelse kan kjennes som sommerfugler i magen, hjertebank og at du
tenker mye på personen."

Omskrevet svar:
"Forelskelse er en ganske spesiell følelse! Det kan kjennes som:
- **Sommerfugler i magen** når du ser personen
- **Hjertebank** og at du blir litt nervøs
- At du **tenker mye** på personen, nesten uten å ville det

Det er helt normalt å kjenne på alt dette!"

========================================
SVAR SOM SKAL OMSKRIVES:
{answer}

========================================
Returner kun det omskrevne svaret. Ingen forklaringer, ingen kommentarer.
"""
)

# GROUNDED_PROMPT = PromptTemplate.from_template(
# """
# Du er en hjelpsom rådgiver og skal svare på norsk (bokmål).

# Du skal returnere:
# - "answer": et samlet svar på spørsmålet.
# - "claims": en liste med påstander med tilhørende citations.
# - "short_answer": en kort oppsummering av "answer"

# ========================================
# REGLER FOR SVAR ("answer")
# ========================================
# - En sammenhengende besvarelse på spørsmålet, skrevet vennlig og tydelig, MEN KUN basert på det som faktisk står i context.
# - Vær vennlig og tydelig.
# - Ikke ta med informasjon som ikke kan støttes direkte av context.
# - Du kan omformulere, men ikke legge til ny informasjon.
# - Ikke ta med informasjon som du ikke kan sitere fra context i etterkant.
# - Ikke referer til context, kilder eller teksten.
# - Bruk markdown-format hvis det forbedrer lesbarheten.
# - Hvis du bruker punktlister/avsnitt: linjeskift må være inne i JSON-strenger (bruk "\\n").

# ========================================
# REGLER FOR CLAIMS
# ========================================
# - En liste av påstander som du har hentet ut fra "answer".
# - Én claim = én tydelig idé.
# - Må være eksplisitt uttrykt eller entydig støttet av context.
# - Ikke finn opp nye ideer.
# - Ikke lag claims basert på spørsmålet eller instruksjonene.


# ========================================
# REGLER FOR CITATIONS
# ========================================
# - Hver citation må være:
#   {{
#     "url": str,
#     "quote": str   # eksakt substring fra context
#   }}
# - "quote" må være ordrett, uten endringer.
# - Ikke slå sammen tekst fra flere steder.
# - Hvis ingen eksakt substring finnes, sett:
#   - "validity": "not valid"
#   - "Citations": []

# ========================================
# SPØRSMÅL:
# {question}

# ========================================
# CONTEXT:
# {context}
# """
# )


# (Optional) A small registry if you prefer string-based lookups
REGISTRY: Dict[str, Prompt] = {
    QA_SUBJECT_NO.id: QA_SUBJECT_NO,
    REFINE_AND_CLASSIFY.id: REFINE_AND_CLASSIFY,
    CLASSIFY_AND_SUBQUERIES.id: CLASSIFY_AND_SUBQUERIES,
    SUBQUERIES.id :SUBQUERIES,
}


# === Helper “renderers” you can import directly ===

def severity_for_query_prompt(query: str) -> str:
    return SEVERITY_FOR_QUERY.render(query=query)

def categorize_text_prompt(text: str, categories: str) -> str:
    return CATEGORIZE_TEXT.render(text=text, categories=categories)

def vectorindex_summary_prompt(text: str) -> str:
    return VECTORINDEX_SUMMARY.render(text=text)

def qa_subject_no_prompt(text: str) -> str:
    return QA_SUBJECT_NO.render(text=text)

def qa_query_rerank_ids_prompt(max_results:str, user_query: str, candidates_jsonl: list) -> str:
    result = QUERY_RERANK_IDS.render( max_results=max_results, user_query=user_query, candidates_jsonl=candidates_jsonl )
    return result

def refine_and_classify_prompt(query: str, categories: str) -> str:
    return REFINE_AND_CLASSIFY.render(query=query, categories=categories)

def classify_and_subqueries_prompt(query: str, categories: str) -> str:
    return CLASSIFY_AND_SUBQUERIES.render(query=query, categories=categories)

def subqueries_prompt(query: str) -> str:
    return SUBQUERIES.render(query=query) 
