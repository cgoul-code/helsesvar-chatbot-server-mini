# query_utils.py

import json

class QuerySettings:
    def __init__(self, **kwargs):
        # Default values provided in the constructor
        self.response_mode = kwargs.get('response_mode', 'tree_summarize')
        self.similarity_top_k = int(kwargs.get('similarity_top_k', 10))  # Default to int, not str
        self.similarity_cutoff = float(kwargs.get('similarity_cutoff', 0.7))  # Default to float
        self.vectorIndex = kwargs.get('vectorIndex', "None")
        self.user_content = kwargs.get('user_content', "")
        self.agent = kwargs.get('agent', "agent_workflow_structured_answer")

    def __str__(self):
        # Convert object properties to a JSON string
        return json.dumps(self.__dict__, ensure_ascii=False, indent=4, default=str)

def get_query_settings(json_request):
    # Use the QuerySettings class constructor with the json_request
    query_settings = QuerySettings(
        response_mode=json_request.get('response_mode', 'tree_summarize'),
        similarity_top_k=json_request.get('similarity_top_k', 10),
        similarity_cutoff=json_request.get('similarity_cutoff', 0.7),
        vectorIndex=json_request.get('vectorIndex', "helsenorgeartikler"),
        agent = json_request.get('agent', "agent_workflow_subquery_orchestrator")
    )
    
    # Messages extraction - more Pythonic way to handle potential missing data
    messages = json_request.get('messages', [])
    query_settings.user_content = next((obj['content'] for obj in messages if obj['role'] == 'user'), None)

    return query_settings
