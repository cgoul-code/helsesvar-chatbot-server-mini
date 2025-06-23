from quart import request, jsonify, Response, send_file
import asyncio 
import logging
import json
from config import server_settings, vector_store, CustomError
from query_utils import get_query_settings
from answer_utils import get_answer_structured_as_stream, get_answer_with_subqueries_as_stream

# --- 1) Definer alle dine agents her ---
# Hver agent må være en async-funksjon som tar de samme argumentene og
# yield’er chunks på SSE-format:
async def stream_with_subqueries(query_settings, server_settings, vector_store):
    async for chunk in get_answer_with_subqueries_as_stream(
        query_settings, server_settings, vector_store
    ):
        yield chunk

async def stream_structured_answer(query_settings, server_settings, vector_store):
    async for chunk in get_answer_structured_as_stream(
        query_settings, server_settings, vector_store
    ):
        yield chunk

# Agent registry: kartlegger agent-navn til funksjon
AGENT_REGISTRY = {
    "subqueries": stream_with_subqueries,
    "structured": stream_structured_answer,
    # legg til flere agenter her…
}
def register_routes(app):

    @app.route("/chat", methods=["POST"])
    async def chat():
        try:
            # --- 2) Sjekk status ---
            status, indexes_loaded = server_settings.get_status()
            if not indexes_loaded:
                logging.warning("Indexes are still loading...")
                logging.info(f"Server status: {status}")
                return {"error": "Indexes are still loading, please try again later."}, 503

            # --- 3) Parse request og settings ---
            json_request = await request.get_json()
            logging.info("Received /chat payload: %r", json_request)
            query_settings = get_query_settings(json_request)

            # --- 4) Velg riktig agent-funksjon ---
            agent_name = query_settings.agent
            agent_fn = AGENT_REGISTRY.get(agent_name)
            if agent_fn is None:
                logging.error(f"Unknown agent requested: {agent_name}")
                return {"error": f"Unknown agent '{agent_name}'"}, 400

            # --- 5) Bygg stream-responsen ---
            async def stream_answer():
                # Vi må formatere hver chunk som SSE:
                async for chunk in agent_fn(query_settings, server_settings, vector_store):
                    # chunk antas å være dict eller ferdig JSON-streng
                    payload = json.dumps(chunk, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
            headers = {
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # (valgfritt bak nginx: "X-Accel-Buffering": "no")
            }

            return Response(stream_answer(), headers=headers)

        except Exception as e:
            logging.exception("Unexpected error in /chat:")
            return {"error": "Internal server error"}, 500