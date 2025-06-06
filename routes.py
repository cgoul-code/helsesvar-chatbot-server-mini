from quart import request, jsonify, Response, send_file
import asyncio 
import logging
import json
from config import server_settings, vector_store, CustomError
from query_utils import get_query_settings
from answer_utils import get_answer_structured, get_answer_with_subqueries_stream

def register_routes(app):

    @app.route("/chat", methods=["POST"])
    async def chat():

        try:
            # Check if indexes are loaded
            status, indexes_loaded = server_settings.get_status()
            if not indexes_loaded:
                logging.warning("Indexes are still loading...")
                logging.info(f'Server status: {status}')
                return {"error": "Indexes are still loading, please try again later."}, 503

            json_request = await request.get_json()
            logging.info("Received /chat payload: %r", json_request)

            query_settings = get_query_settings(json_request)

            if query_settings.agent == "agent_workflow_structured_answer":
                answer = get_answer_structured(query_settings, server_settings, vector_store)
                return {"answer": answer}, 200

            if query_settings.agent == "agent_workflow_subquery_orchestrator":
                async def stream_answer():
                    async for chunk in get_answer_with_subqueries_stream(
                        query_settings, server_settings, vector_store
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"

                return Response(stream_answer(), content_type="text/event-stream")

            else:
                raise CustomError(f"Agent {query_settings.agent} mangler!", 404)

        except Exception as e:
            logging.error("Error in /chat handler", exc_info=True)
            status = getattr(e, "code", 500)
            return {"error": str(e)}, status
