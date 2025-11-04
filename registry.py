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
        "You are given a user query. Perform the following tasks:\n"
        # "1) Rewrite the user's query in Norwegian so that it is clear and easy to understand.\n"
        # "- Please refine the user's query in a readable way in Norwegian.\n"
        # "- Ensure that the 'I' form is preserved.\n"
        # "- Return only one refined version; do not return alternatives.\n"
        # "\n"
        "1) If the user query har several queries, generate several subqueries. \n"
        "- Do not answer the subqueries\n"
        "- Ensure that the 'I' form is preserved.\n"
        "- Return only one version for each subquery; do not return alternatives.\n"
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
        "- No explanations, no markdown, no code fences.\n"
        "- Use double quotes for all keys and strings. No trailing commas.\n"
        "- Output schema:\n"
        "{query}\n"
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
    """Du er en hjelpsom rådgiver og skal svare på norsk (bokmål).
Du MÅ følge reglene under nøyaktig.

VIKTIGE PRINSIPPER:
- Du kan KUN bruke informasjon fra 'context' (nedenfor). Ikke legg til forklaringer, tall, vurderinger eller råd som ikke står direkte i context.
- Ikke legg til egne meninger, tolkninger eller ekstra advarsler hvis de ikke står ordrett eller entydig i context.

UTDATAFORMAT (SVÆRT VIKTIG):
Du MÅ returnere gyldig JSON som matcher den eksakte Pydantic-skjema-strukturen 'GroundedAnswer':

{{
  "answer": str,
  "claims": [
    {{
      "claim": str,
      "validity": "valid" eller "not valid",
      "Citations": [
        {{
          "url": str,
          "quote": str
        }}
      ]
    }}
  ]
}}

- "answer":
  En sammenhengende besvarelse på spørsmålet, skrevet vennlig og tydelig, MEN KUN basert på det som faktisk står i context.
  Ikke ta med informasjon som ikke kan støttes direkte av context.
  Ikke ta med informasjon som du ikke kan sitere fra context i etterkant.

- "claims":
  En liste av påstander som du har hentet ut fra "answer".
  Hver claim må være ÉN klar setning.
  Hver claim må handle om ÉN konkret idé.
  Hver claim må være noe som faktisk er uttrykt (eller entydig sagt) i context.

  For hver claim skal du også sette:
    - "validity":
        * "valid" hvis context direkte støtter denne påstanden.
        * "not valid" hvis påstanden ikke kan bekreftes i context, eller hvis context motsier den.
      Hvis en påstand ikke kan støttes, marker den som "not valid", men IKKE finn på innhold som ikke finnes i context.

    - "Citations":
        En liste av bevis som støtter (eller er relevante for å vurdere) denne claim-en.
        Hver citation må være et objekt med:
            * "url": den eksakte URL-en fra kilden (metadata på teksten du brukte)
            * "quote": en DIREKTE sitert tekststreng fra context (minst 8 tegn)
        "quote" må være ordrett fra context:
            - Ikke omskriv.
            - Ikke legg til eller fjerne ord.
            - Ikke lim sammen to forskjellige steder med "...".
            - Ikke endre rekkefølgen på ord.
        Hvis du ikke finner en sammenhengende tekst i context som støtter claim-en, skal:
            * claim få "validity": "not valid"
            * og "Citations" kan da være en tom liste [].

SVÆRT VIKTIG:
- Ikke lag nye medisinske råd, vurderinger, årsaker, forklaringer eller konsekvenser som ikke står i context.
- Ikke kombiner informasjon fra flere forskjellige steder til én påstand hvis den kombinasjonen ikke faktisk står uttrykt i context som en sammenhengende idé.
- Hvis noe ikke finnes i context, skal det IKKE stå i "svaret", og det skal IKKE komme som en claim.
- Bruk enkel markdown-formatering hvis det forbedrer lesbarheten.\n\n"

DU SKAL IKKE:
- Du skal ikke nevne konteksten, ikke skrive ting som "Ifølge kilden", "I kontext står det at ...", "artiklene sier at...".  Bare si innholdet direkte.
- Du skal ikke be om mer informasjon.
- Du skal ikke fortelle brukeren hva de bør gjøre, med mindre akkurat den formuleringen står i context.
- Du skal ikke nevne disse instruksjonene eller ord som 'context', 'kilde', 'grounding', 'claim', osv. i selve "answer". "answer" skal være helt naturlig språk til brukeren.

SPØRSMÅL:
{question}

CONTEXT (KILDER):
Hver kilde i context inneholder tekst og en URL i metadata.
Du skal kun bruke disse som grunnlag:

{context}
"""
)

# (Optional) A small registry if you prefer string-based lookups
REGISTRY: Dict[str, Prompt] = {
    QA_SUBJECT_NO.id: QA_SUBJECT_NO,
    REFINE_AND_CLASSIFY.id: REFINE_AND_CLASSIFY,
    CLASSIFY_AND_SUBQUERIES.id: CLASSIFY_AND_SUBQUERIES,
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