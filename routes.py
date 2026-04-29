from quart import request, Response
import logging, json, asyncio
from typing import Any, Dict, List, Optional, Tuple
from config import server_settings, vector_store
import secrets
import diskcache
from query_utils import get_query_settings
from answer_utils import (
    get_answer_as_stream, get_related_qa_as_stream, get_examples_full_as_stream
)

AGENT_REGISTRY = {
    "hvaerinnafor": get_answer_as_stream,
    "hvaerinnafor_related_qa": get_related_qa_as_stream,
    "hvaerinnafor_examples": get_examples_full_as_stream, 
}

SESSION_STORE = diskcache.Cache("./session_cache")


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
    def _cors_preflight():
        return Response(
            "",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type, Accept",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
            },
        )

    def _not_ready_response(status: str) -> Response:
        """503 response the browser can actually read (explicit CORS + JSON)."""
        body = {
            "error": "server_not_ready",
            "ready": False,
            "status": status,
            "message": "Serveren laster fortsatt indekser. Prøv igjen om noen sekunder.",
        }
        return Response(
            json.dumps(body, ensure_ascii=False),
            status=503,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Access-Control-Allow-Origin": "*",
                "Retry-After": "5",
            },
        )

    def _is_duplicate_last(history: List[Dict[str, str]], msg: Dict[str, str]) -> bool:
        """Return True hvis msg er identisk med siste element i history."""
        if not history:
            return False
        last = history[-1]
        return (
            (last.get("role") == msg.get("role"))
            and (last.get("content") == msg.get("content"))
        )

    def _get_or_create_session_id(query_settings, payload: Dict[str, Any]) -> str:
        session_id = getattr(query_settings, "session_id", None) or payload.get("session_id")
        if not session_id:
            session_id = secrets.token_hex(8)
        query_settings.session_id = session_id
        return session_id

    def _load_history(session_id: str) -> List[Dict[str, str]]:
        # copy for å unngå utilsiktet mutasjon via referanser
        return list(SESSION_STORE.get(session_id, []))

    def _extract_last_user_message(payload: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Vi antar payload["messages"] kan være:
          - hele historikken
          - bare nye meldinger
          - bare siste user
        Vi vil KUN ta med siste user-melding her.
        """
        msgs = payload.get("messages", [])
        if not isinstance(msgs, list) or not msgs:
            return None

        last = msgs[-1]
        if not isinstance(last, dict):
            return None

        if last.get("role") != "user":
            # Hvis klient sender både user+assistant, kan siste være assistant.
            # Finn siste user bakover.
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") == "user":
                    return m
            return None

        return last
    
    MAX_HISTORY_MESSAGES = 20      # maks antall meldinger (user + assistant)
    MAX_HISTORY_CHARS    = 8_000   # maks tegn total i historikken


    def _trim_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Kutter historikken fra eldste ende til den er innenfor begge grensene.
        Bevarer alltid par (user → assistant) for å unngå at historikken
        starter midt i en utveksling.
        """
        # 1) Kapp på antall meldinger
        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[-MAX_HISTORY_MESSAGES:]

        # 2) Kapp på totalt antall tegn
        while history:
            total_chars = sum(len(m.get("content", "")) for m in history)
            if total_chars <= MAX_HISTORY_CHARS:
                break
            # Fjern eldste melding, men behold alltid minst én user+assistant-runde
            if len(history) <= 2:
                break
            history = history[1:]

        # 3) Sørg for at historikken starter med en user-melding (ikke et halvt par)
        while history and history[0].get("role") != "user":
            history = history[1:]

        return history

    def _store_user_message(
        session_id: str,
        history: List[Dict[str, str]],
        user_msg: Optional[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        if user_msg and user_msg.get("content"):
            if not _is_duplicate_last(history, user_msg):
                history.append({"role": "user", "content": user_msg["content"]})

        # Trim før lagring så SESSION_STORE aldri vokser ubegrenset
        history = _trim_history(history)
        SESSION_STORE[session_id] = history
        return history

    def _store_assistant_message(
        session_id: str,
        history: List[Dict[str, str]],
        assistant_text: str,
    ) -> List[Dict[str, str]]:
        assistant_text = (assistant_text or "").strip()
        if assistant_text:
            msg = {"role": "assistant", "content": assistant_text}
            if not _is_duplicate_last(history, msg):
                history.append(msg)

        # Trim etter assistant også — et langt svar kan alene overskride grensen
        history = _trim_history(history)
        SESSION_STORE[session_id] = history
        return history
    
    @app.route("/healthz", methods=["GET", "OPTIONS"])
    async def healthz():
        """Lightweight readiness probe for clients to poll before calling /chat or /examples.

        Always returns 200 — the body's `ready` flag tells the client whether the
        backend has finished loading indexes. (Returning 200 here keeps the polling
        loop simple; the chat/examples endpoints still return 503 for safety.)
        """
        if request.method == "OPTIONS":
            return _cors_preflight()

        status, indexes_loaded = server_settings.get_status()
        body = {
            "ready": bool(indexes_loaded),
            "status": status,
        }
        if not indexes_loaded:
            body["message"] = "Serveren laster fortsatt indekser. Prøv igjen om noen sekunder."

        return Response(
            json.dumps(body, ensure_ascii=False),
            status=200,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-store",
            },
        )

    @app.route("/examples", methods=["POST", "OPTIONS"])
    async def examples():
        if request.method == "OPTIONS":
            return _cors_preflight()

        # Index readiness
        status, indexes_loaded = server_settings.get_status()
        if not indexes_loaded:
            logging.warning("Indexes are still loading (status=%s)", status)
            return _not_ready_response(status)

        try:
            payload = await request.get_json()
            query_settings = get_query_settings(payload)

            agent_name = getattr(query_settings, "agent", None) or "hvaerinnafor_examples"
            agent_fn = AGENT_REGISTRY.get(agent_name)
            if agent_fn is None:
                return {"error": f"Unknown agent '{agent_name}'"}, 400

            async def stream_examples():
                yield _format_sse(json.dumps({"event": "open", "message": "ok"}, ensure_ascii=False))
                try:
                    async for chunk in agent_fn(query_settings, server_settings, vector_store):
                        data = json.dumps(chunk, ensure_ascii=False) if isinstance(chunk, dict) else str(chunk)
                        yield _format_sse(data)
                        await asyncio.sleep(0)

                except (asyncio.CancelledError, GeneratorExit):
                    return
                except Exception as e:
                    logging.error("Error while streaming examples output", exc_info=True)
                    yield _format_sse(json.dumps({"event": "error", "error": str(e)}, ensure_ascii=False))
                finally:
                    # make sure done always comes
                    yield _format_sse(json.dumps({"event": "done"}, ensure_ascii=False))

            headers = {
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
            }
            return Response(stream_examples(), headers=headers)

        except Exception as e:
            logging.error("Error in /examples handler", exc_info=True)
            status_code = getattr(e, "code", 500)
            return {"error": str(e)}, status_code

    

    @app.route("/chat", methods=["POST", "OPTIONS"])
    async def chat():
        # CORS preflight
        if request.method == "OPTIONS":
            return _cors_preflight()

        # Index readiness
        status, indexes_loaded = server_settings.get_status()
        if not indexes_loaded:
            logging.warning("Indexes are still loading (status=%s)", status)
            return _not_ready_response(status)

        try:
            payload = await request.get_json()
            logging.info("Received /chat payload: %r", payload)

            query_settings = get_query_settings(payload)

            # 1) resolve / create session_id
            session_id = _get_or_create_session_id(query_settings, payload)

            # 2) load existing history
            history = _load_history(session_id)

            # 3) append ONLY the latest user message from payload
            last_user_msg = _extract_last_user_message(payload)
            history = _store_user_message(session_id, history, last_user_msg)

            # 4) pass full server-side history into agent
            query_settings.messages = history

            # Agent resolution
            agent_name = getattr(query_settings, "agent", None)
            agent_fn = AGENT_REGISTRY.get(agent_name)
            if agent_fn is None:
                logging.error("Unknown agent requested: %s", agent_name)
                return {"error": f"Unknown agent '{agent_name}'"}, 400

            async def stream_answer():
                yield _format_sse(json.dumps({"event": "open", "message": "ok", "session_id": session_id}, ensure_ascii=False))

                heartbeat_interval = 20
                last_sent = asyncio.get_event_loop().time()

                # IMPORTANT: hent fersk historikk (copy) for denne streamen
                stream_history = _load_history(session_id)

                assistant_buffer: List[str] = []

                try:
                    async for chunk in agent_fn(query_settings, server_settings, vector_store):
                        # chunk er ofte dict-event fra workflowen
                        if isinstance(chunk, dict):
                            if chunk.get("event") == "answer":
                                delta = (
                                    chunk.get("structured_answer_delta")
                                    or chunk.get("message")
                                    or ""
                                )
                                if delta:
                                    assistant_buffer.append(delta)

                            data = json.dumps(chunk, ensure_ascii=False)
                        else:
                            data = str(chunk)

                        yield _format_sse(data)

                        # heartbeat bookkeeping
                        last_sent = asyncio.get_event_loop().time()
                        await asyncio.sleep(0)

                        now = asyncio.get_event_loop().time()
                        if now - last_sent >= heartbeat_interval:
                            yield ": ping\n\n"
                            last_sent = now

                except (asyncio.CancelledError, GeneratorExit):
                    return
                except Exception as e:
                    logging.error("Error while streaming agent output", exc_info=True)
                    yield _format_sse(json.dumps({"event": "error", "error": str(e)}, ensure_ascii=False))
                finally:
                    # 5) store assistant full answer ONCE
                    full_answer = "".join(assistant_buffer).strip()
                    if session_id:
                        # hent siste historikk igjen i tilfelle andre forespørsler har skrevet
                        final_history = _load_history(session_id)
                        _store_assistant_message(session_id, final_history, full_answer)

                    yield _format_sse(json.dumps({"event": "done"}, ensure_ascii=False))

            headers = {
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
            }
            return Response(stream_answer(), headers=headers)

        except Exception as e:
            logging.error("Error in /chat handler", exc_info=True)
            status_code = getattr(e, "code", 500)
            return {"error": str(e)}, status_code