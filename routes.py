from quart import request, jsonify, Response
import logging
from config import (server_settings, vector_store, CustomError)
from query_utils import (get_query_settings)
from answer_utils import (get_structured_answer)

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
            # get info from query
            #
            query_settings = get_query_settings(json_request)
            
            # route to the correct agent
            #
            if(query_settings.agent == "agent_workflow_structured_answer"):
                answer = get_structured_answer(query_settings, server_settings, vector_store)
                return {"answer": answer}, 200
            else:
                raise CustomError(
                    f"Agent {query_settings.agent} mangler!", 404
        )
        
        except Exception as e:
            logging.error("Error in /chat handler", exc_info=True)
            status = getattr(e, "code", 500)          # default to 500 if no .code
            return {"error": str(e)}, status