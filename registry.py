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

GROUNDED_PROMPT_old = PromptTemplate.from_template(
"""
Du er en hjelpsom rådgiver og skal svare på norsk (bokmål).

Du skal fylle ut følgende felter:
- "answer": et samlet svar på spørsmålet.
- "claims": en liste med påstander med tilhørende citations.

Du MÅ følge reglene under nøyaktig og uten unntak.

========================================
VIKTIGE PRINSIPPER
========================================
- Du kan KUN bruke informasjon fra "context" nedenfor.
- Ikke legg til nye forklaringer, tall, vurderinger eller råd som ikke står ordrett eller entydig i context.
- Ikke gjør egne tolkninger eller gjetninger.
- Ikke omskriv eller omformuler tekst i feltet "quote" i citations.
- Hvis context ikke støtter et utsagn, skal det IKKE stå i "answer".

========================================
REGLER FOR "answer"
========================================
- "answer" skal være et vennlig, tydelig og sammenhengende svar på norsk (bokmål).
- Du skal forklare med egne ord, men KUN basert på informasjon som faktisk står i context.
- Du kan omformulere tekst fra context i "answer" så lenge innholdet ikke endres.
- Ikke inkluder informasjon, vurderinger, råd eller årsaker som ikke står eksplisitt i context.
- Ikke nevn "context", "kilder", "artikler", "teksten" eller lignende.

========================================
REGLER FOR "claims"
========================================
- En claim er ÉN klar setning som uttrykker EN idé.
- Alle claims må være eksplisitt uttrykt eller entydig formulert i context.
- Ikke konstruer kombinasjoner av informasjon som ikke står slik i context som én sammenhengende idé.
- Ikke lag claims som bare gjengir brukerens spørsmål eller instruksjonsteksten.
- Claims skal kun beskrive innhold fra context som er relevant for spørsmålet.

For hver claim:
- "validity" = "valid" hvis context direkte støtter hele claim-en som en hel setning eller idé.
- "validity" = "not valid" hvis context ikke støtter den, eller motsier den.
- Ikke legg til forklaringer som ikke står i context.

========================================
KRAV TIL "citations" (SVÆRT VIKTIG)
========================================
En citation består av:
{{
  "url": str,     # identisk med URL-en i metadata
  "quote": str    # eksakt substring fra context
}}

REGLER for citations:
- "quote" MÅ være en eksakt, uendret substring hentet direkte fra context.
- Kun "quote" har kravet om å være ordrett; "answer" kan være lett omformulert.
- Ikke endre bokstaver, mellomrom, linjeskift, tegnsetting eller rekkefølge.
- Ikke bruk staveendringer.
- Ikke bruk unicode-varianter.
- Ikke slå sammen tekst fra to steder.
- Ikke inkluder byte-korrupt tekst eller genererte kontrolltegn.
- Hvis du ikke finner en eksakt substring som støtter claim-en, skal:
    * "validity" = "not valid"
    * "Citations" = []

========================================
TEKNISKE SIKKERHETSREGLER (NYTTIGE)
========================================
- Reglene over om eksakt tekst gjelder KUN for "quote" i citations, ikke for "answer".
- Du har ikke lov til å "forbedre" sitater i "quote".
- Hvis du er i tvil om et sitat er eksakt, skal du heller la "Citations" være tomt.

========================================
SPØRSMÅL:
{question}

========================================
CONTEXT (KILDER):
Hver kilde i context inneholder tekst og en URL i metadata.
Du skal kun bruke disse som grunnlag:

{context}
"""
)

GROUNDED_PROMPT = PromptTemplate.from_template(
"""
Du er en hjelpsom rådgiver og skal svare på norsk (bokmål).

Du skal returnere:
- "answer": et samlet svar på spørsmålet.
- "claims": en liste med påstander med tilhørende citations.


========================================
REGLER FOR SVAR ("answer")
========================================
- Svar kun basert på det som står i context.
- Vær vennlig og tydelig.
- Du kan omformulere, men ikke legge til ny informasjon.
- Ikke referer til context, kilder eller teksten.

========================================
REGLER FOR CLAIMS
========================================
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
"""
)


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
