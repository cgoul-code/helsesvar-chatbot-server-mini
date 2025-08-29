from quart import request, jsonify, Response
import logging
from config import (server_settings, vector_store)
from query_utils import (get_query_settings)
from answer_utils import (get_answer)

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
            answer = get_answer(query_settings, server_settings, vector_store)
            return {"answer": answer}, 200
        
        except Exception as e:
            logging.error("Error in /chat handler", exc_info=True)
            status = getattr(e, "code", 500)          # default to 500 if no .code
            return {"error": str(e)}, status