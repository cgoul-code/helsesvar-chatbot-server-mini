# prompts/registry.py
from dataclasses import dataclass
from typing import Dict

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



REFINE_QUERY = Prompt(
    id="refine_query",
    template=(
        "You are given a text: {text}\n"
        "Please refine the user's query in readable way in norwegian.\n"
        "Ensure that the 'I' form is preserved.\n"
        "Return only one refined, do not return alternatives\n"
    )
)


REFINE_QUERY = Prompt(
    id="refine_query",
    template=(
        "You are given a user query: {query}\n"
        "Please refine the user's query in readable way in norwegian.\n"
        "Ensure that the 'I' form is preserved.\n"
        "Return only one refined, do not return alternatives\n"
    )
)


# === TEMPLATES ===
SEVERITY_FOR_TEXT = Prompt(
    id="severity_for_text",
    template=(
        "You are given a text: {text}\n\n"
        "Categorize the severity of this text into one of three categories: Green, Yellow, or Red.\n\n"
        "Green category:\n"
        "- Preventive and safety-promoting.\n"
        "- Texts that provide general information, knowledge, and guidance to prevent problems and strengthen good sexual health.\n"
        "- The content helps increase understanding, safety, and awareness (e.g., consent, contraception, communication, emotions, body knowledge).\n"
        "- No acute situation or personal crisis is described.\n"
        'Example: "How to talk with your partner about boundaries" or "Facts about condoms".\n\n'
        "Yellow category:\n"
        "- Challenges or vulnerable situations.\n"
        "- Texts that describe concerns, difficulties, or risks that may require reflection or support, but are not acute or immediately dangerous.\n"
        "- May involve difficult feelings, uncertainty in relationships, unwanted experiences, or the need for advice beyond general information.\n"
        "- The reader may need to seek help or guidance, but the situation is not considered an acute crisis.\n"
        'Example: "What should I do if my partner doesn’t respect my boundaries?", '
        '"I regret sending a nude", or topics like "pornography", "sexual pressure", "issues around consent", "(illegal) fetishes".\n\n'
        "Red category:\n"
        "- Serious or acute situations.\n"
        "- Texts that concern serious incidents or crises where the person involved may be in danger or at significant risk of harm.\n"
        "- Includes violence, abuse, coercion, acute psychological crises, or other situations that require immediate follow-up or professional help.\n"
        "- The main purpose of the text is to provide information about where and how to get help quickly.\n"
        'Example: "Sex with animals", "Sex with family members", "Downloading child pornography", "Illegal image sharing".\n\n'
        "Output requirements:\n"
        "- Return JSON ONLY. No explanations. No markdown. No code fences.\n"
        '- Use double quotes for all keys and strings.\n'
        "- Do not include trailing commas.\n"
        '- Output shape:\n'
        '{{ "category": "<Green|Yellow|Red>" }}'
    ),
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

# (Optional) A small registry if you prefer string-based lookups
REGISTRY: Dict[str, Prompt] = {
    SEVERITY_FOR_TEXT.id: SEVERITY_FOR_TEXT,
    SEVERITY_FOR_QUERY.id: SEVERITY_FOR_QUERY,
    QA_SUBJECT_NO.id: QA_SUBJECT_NO,
}


# === Helper “renderers” you can import directly ===
def vectorindex_summary_prompt(text: str) -> str:
    return VECTORINDEX_SUMMARY.render(text=text)

def refine_query_prompt(query: str) -> str:
    return REFINE_QUERY.render(query=query)

def severity_for_text_prompt(text: str) -> str:
    return SEVERITY_FOR_TEXT.render(text=text)

def severity_for_query_prompt(query: str) -> str:
    return SEVERITY_FOR_QUERY.render(query=query)

def qa_subject_no_prompt(text: str) -> str:
    return QA_SUBJECT_NO.render(text=text)

def qa_query_rerank_ids_prompt(max_results:str, user_query: str, candidates_jsonl: list) -> str:
    result = QUERY_RERANK_IDS.render( max_results=max_results, user_query=user_query, candidates_jsonl=candidates_jsonl )
    return result
