from quart import request, jsonify, Response
import logging
from config import (server_settings, vector_store)
from query_utils import (get_query_settings)
from answer_utils import get_structured_answer, get_answer_with_related_queries

# Agent registry: kartlegger agent-navn til funksjon
AGENT_REGISTRY = {
    "hvaerinnafor": get_answer_with_related_queries,
    "structured": get_structured_answer,
    # legg til flere agenter her…
}

def register_routes(app):
    @app.route("/chat", methods=["POST"])
    async def chat():
        # Check if indexes are loaded
        status, indexes_loaded = server_settings.get_status()
        if not indexes_loaded:
            logging.warning("Indexes are still loading...")
            logging.info(f'Server status: {status}')
            return {"error": "Indexes are still loading, please try again later."}, 503
        
        try:
            json_request = await request.get_json()
            logging.info("Received /chat payload: %r", json_request)
            # your real logic here...
            
            query_settings = get_query_settings(json_request)
            
              # --- 4) Velg riktig agent-funksjon ---
            agent_name = query_settings.agent
            agent_fn = AGENT_REGISTRY.get(agent_name)
            print(f'------->(name:{agent_name}, fn:{agent_fn})<--------------')
            if agent_fn is None:
                logging.error(f"Unknown agent requested: {agent_name}")
                return {"error": f"Unknown agent '{agent_name}'"}, 400
            
            answer = agent_fn(query_settings, server_settings, vector_store)
            return {"answer": answer}, 200
        
        except Exception as e:
            logging.error("Error in /chat handler", exc_info=True)
            status = getattr(e, "code", 500)          # default to 500 if no .code
            return {"error": str(e)}, status