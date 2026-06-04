# query_utils.py

import json

class QuerySettings:
    def __init__(self, **kwargs):
        # Default values provided in the constructor
        self.response_mode = kwargs.get('response_mode', 'tree_summarize')
        self.similarity_top_k = int(kwargs.get('similarity_top_k', 10))  # Default to int, not str
        self.similarity_cutoff = float(kwargs.get('similarity_cutoff', 0.45))  # Default to float
        self.relevancy_cutoff = float(kwargs.get('relevancy_cutoff', 0.45))  # Default to float
        self.vectorIndex = kwargs.get('vectorIndex', "None")
        self.qa_bank_index = kwargs.get('qa_bank_index', None)  # explicit QA-bank override; None = use fallback chain
        # Optional client override for response style. "" = auto-route via
        # pick_response_style(severity, stance). Valid values: factual,
        # warm, supportive, crisis. Unknown values fall back to auto.
        self.response_style = kwargs.get('response_style', "")
        self.user_content = kwargs.get('user_content', "")
        
        self.agent = kwargs.get('agent', "structured")
        self.related_only = kwargs.get('related_only', False)  
        self.main_category  = kwargs.get('main_category', "")  
        self.query_severity  = kwargs.get('query_severity', "")  
        self.query = kwargs.get('query', "")  
        self.from_related_q = kwargs.get('from_related_q', False)  
        self.from_node_id = kwargs.get('from_node_id', "")
        self.requested_categories = kwargs.get('requested_categories', [])  
        
        self.session_id = kwargs.get('session_id', None)
        self.messages = kwargs.get('messages', [])
        # Min fraction of cited claims that must be supported for an answer to
        # stay valid in query_grounded. Default 1.0.
        self.claims_valid_threshold = float(kwargs.get('claims_valid_threshold', 1.0))
        # When True, query_grounded runs an LLM entailment gate that drops
        # claims whose (real) quote doesn't support them. Default True.
        self.entailment_check = bool(kwargs.get('entailment_check', True))
        # Debug only: when True, query_grounded emits the exact retrieved nodes
        # (text + score) as a `retrieved_nodes` SSE event. Off in production.
        self.debug_emit_nodes = bool(kwargs.get('debug_emit_nodes', False))

    def __str__(self):
        # Convert object properties to a JSON string
        return json.dumps(self.__dict__, ensure_ascii=False, indent=4, default=str)

def get_query_settings(json_request):
    # Use the QuerySettings class constructor with the json_request
    query_settings = QuerySettings(
        response_mode=json_request.get('response_mode', 'tree_summarize'),
        similarity_top_k=json_request.get('similarity_top_k', 10),
        similarity_cutoff=json_request.get('similarity_cutoff', 0.45),
        relevancy_cutoff=json_request.get('relevancy_cutoff', 0.45),
        vectorIndex=json_request.get('vectorIndex', "hvaerinnafor"),
        qa_bank_index=json_request.get('qa_bank_index', None),
        response_style=json_request.get('response_style', ""),
        agent = json_request.get('agent', "structured"),
        related_only = json_request.get('related_only', False),  
        main_category = json_request.get('main_category', ""),  
        query_severity = json_request.get('query_severity', "") , 
        query = json_request.get('query', ""),
        from_related_q = json_request.get('from_related_q', False),
        from_node_id = json_request.get('from_node_id', ""),
        requested_categories = json_request.get('requested_categories', []),
        claims_valid_threshold = json_request.get('claims_valid_threshold', 1.0),
        entailment_check = json_request.get('entailment_check', True),

        session_id=json_request.get('session_id'),
        messages=json_request.get('messages', []),

    )
    
    # Messages extraction - more Pythonic way to handle potential missing data
    messages = json_request.get('messages', [])
    query_settings.user_content = next((obj['content'] for obj in messages if obj['role'] == 'user'), None)

    return query_settings