from quart import request, Response
import logging, json, asyncio
from config import server_settings, vector_store
from query_utils import get_query_settings
from answer_utils import (
    get_answer_as_stream, get_related_qa_as_stream
)

# Agents must be async generators that yield small dict/str chunks
AGENT_REGISTRY = {
    "hvaerinnafor": get_answer_as_stream,
    "hvaerinnafor_related_qa": get_related_qa_as_stream,
#    "structured": get_answer_structured_as_stream,
}

SESSION_STORE: dict[str, list[dict]] = {}

def _format_sse(data: str, event: str | None = None) -> str:
    """Format one SSE message (optionally named), ending with a blank line."""
    lines = []
    if event:
        lines.append(f"event: {event}")
    for line in (data.splitlines() or [""]):
        lines.append(f"data: {line}")
    lines.append("")  # blank line terminator
    return "\n".join(lines)

def register_routes(app):
    @app.route("/chat", methods=["POST", "OPTIONS"])
    async def chat():
        # Simple CORS (adjust as needed)
        if request.method == "OPTIONS":
            return Response(
                "",
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type, Accept",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                },
            )

        # Index readiness
        status, indexes_loaded = server_settings.get_status()
        if not indexes_loaded:
            logging.warning("Indexes are still loading...")
            logging.info(f"Server status: {status}")
            return {"error": "Indexes are still loading, please try again later."}, 503

        try:
            payload = await request.get_json()
            logging.info("Received /chat payload: %r", payload)

            query_settings = get_query_settings(payload)
            
            # 1) resolve / create session_id
            session_id = query_settings.session_id or payload.get("session_id")
            if not session_id:
                # very simple: generate one (in prod you’d use uuid4)
                import secrets
                session_id = secrets.token_hex(8)
            query_settings.session_id = session_id

            # 2) load existing history or start fresh
            history = SESSION_STORE.get(session_id, [])

            # 3) merge client messages into history:
            #    (here I assume payload.messages are the new messages since last time,
            #     or you can trust them fully and just store them)
            new_messages = payload.get("messages", [])
            history.extend(new_messages)
            SESSION_STORE[session_id] = history

            # 4) pass full history into agent via query_settings
            query_settings.messages = history
                
            agent_name = getattr(query_settings, "agent", None)
            agent_fn = AGENT_REGISTRY.get(agent_name)
            if agent_fn is None:
                logging.error("Unknown agent requested: %s", agent_name)
                return {"error": f"Unknown agent '{agent_name}'"}, 400

            async def stream_answer():
                yield _format_sse(json.dumps({"event": "open", "message": "ok"}, ensure_ascii=False))

                heartbeat_interval = 20
                last_sent = asyncio.get_event_loop().time()

                # NEW: get session + history (assuming you added it earlier)
                session_id = getattr(query_settings, "session_id", None)
                history = SESSION_STORE.get(session_id, [])

                assistant_buffer: list[str] = []

                try:
                    async for chunk in agent_fn(query_settings, server_settings, vector_store):
                        # chunk is usually a dict from the workflow
                        if isinstance(chunk, (dict, list)):
                            # NEW: inspect internal "event"
                            if isinstance(chunk, dict) and chunk.get("event") == "answer":
                                delta = (
                                    chunk.get("structured_answer_delta")
                                    or chunk.get("message")
                                    or ""
                                )
                                assistant_buffer.append(delta)

                            data = json.dumps(chunk, ensure_ascii=False)
                        else:
                            data = str(chunk)

                        yield _format_sse(data)
                        last_sent = asyncio.get_event_loop().time()

                        await asyncio.sleep(0)

                        now = asyncio.get_event_loop().time()
                        if now - last_sent >= heartbeat_interval:
                            yield ": ping\n\n"
                            last_sent = now

                except (asyncio.CancelledError, GeneratorExit):
                    return
                except Exception as e:
                    err = {"error": str(e)}
                    yield _format_sse(json.dumps({"event": "error", **err}, ensure_ascii=False))
                finally:
                    # NEW: when answer is done, store it in history
                    full_answer = "".join(assistant_buffer).strip()
                    if session_id and full_answer:
                        history.append({"role": "assistant", "content": full_answer})
                        SESSION_STORE[session_id] = history

                    # done event
                    yield _format_sse(json.dumps({"event": "done"}, ensure_ascii=False))

            headers = {
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # important behind nginx
                "Access-Control-Allow-Origin": "*",  # if you need CORS
            }
            return Response(stream_answer(), headers=headers)

        except Exception as e:
            logging.error("Error in /chat handler", exc_info=True)
            status = getattr(e, "code", 500)
            return {"error": str(e)}, status
        
        
