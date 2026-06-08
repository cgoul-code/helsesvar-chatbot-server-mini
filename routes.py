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


# SSE comment line — silently ignored by EventSource clients but keeps the
# socket alive across browser/proxy/Hypercorn idle timeouts.
SSE_HEARTBEAT = ": keepalive\n\n"
HEARTBEAT_INTERVAL_S = 15


async def _with_heartbeat(agen, interval: float = HEARTBEAT_INTERVAL_S):
    """Wrap an async generator and emit ('heartbeat', None) when it's idle.

    Yields tuples of:
      ("chunk", item)      — a real value produced by `agen`
      ("heartbeat", None)  — emitted every `interval` seconds of silence

    Exceptions raised by `agen` propagate out of the consumer's `async for`.
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def _pump():
        try:
            async for item in agen:
                await queue.put(("chunk", item))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # propagate to consumer
            await queue.put(("error", e))
        finally:
            await queue.put(("done", None))

    pump_task = asyncio.create_task(_pump())
    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=interval)
            except asyncio.TimeoutError:
                yield ("heartbeat", None)
                continue

            if kind == "chunk":
                yield ("chunk", payload)
            elif kind == "error":
                raise payload
            elif kind == "done":
                return
    finally:
        if not pump_task.done():
            pump_task.cancel()
            try:
                await pump_task
            except BaseException:
                pass

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

    @app.route("/categories", methods=["GET", "OPTIONS"])
    async def categories():
        """Return the sorted distinct categories present in the hvaerinnafor index."""
        if request.method == "OPTIONS":
            return _cors_preflight()

        status, indexes_loaded = server_settings.get_status()
        if not indexes_loaded:
            logging.warning("Indexes are still loading (status=%s)", status)
            return _not_ready_response(status)

        seen: set[str] = set()
        entry = vector_store.get("hvaerinnafor")
        if entry is not None:
            for node in entry.index.docstore.docs.values():
                cats = (getattr(node, "metadata", None) or {}).get("categories")
                if isinstance(cats, list):
                    for c in cats:
                        c = str(c).strip()
                        if c:
                            seen.add(c)

        return Response(
            json.dumps(sorted(seen), ensure_ascii=False),
            status=200,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-store",
            },
        )

    @app.route("/documents", methods=["GET", "OPTIONS"])
    async def documents():
        """Return the distinct source documents loaded in the hvaerinnafor index.

        Deduplicates the docstore nodes by URL (used as doc_id at ingest time) and
        returns one entry per source document with its title, category and
        categories. Sorted by title for stable client rendering.
        """
        if request.method == "OPTIONS":
            return _cors_preflight()

        status, indexes_loaded = server_settings.get_status()
        if not indexes_loaded:
            logging.warning("Indexes are still loading (status=%s)", status)
            return _not_ready_response(status)

        by_url: Dict[str, Dict[str, Any]] = {}
        entry = vector_store.get("hvaerinnafor")
        if entry is not None:
            for node in entry.index.docstore.docs.values():
                meta = getattr(node, "metadata", None) or {}
                url = (meta.get("url") or "").strip()
                if not url or url in by_url:
                    continue
                cats = meta.get("categories")
                by_url[url] = {
                    "url": url,
                    "title": (meta.get("title") or "").strip(),
                    "category": (meta.get("category") or "").strip(),
                    "categories": cats if isinstance(cats, list) else [],
                    "description": (meta.get("description") or "").strip(),
                }

        docs = sorted(by_url.values(), key=lambda d: d["title"].casefold())
        body = {"count": len(docs), "documents": docs}

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
                chunks_sent = 0
                agent_completed = False
                client_disconnected = False
                try:
                    async for kind, item in _with_heartbeat(
                        agent_fn(query_settings, server_settings, vector_store)
                    ):
                        if kind == "heartbeat":
                            yield SSE_HEARTBEAT
                            continue
                        data = json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
                        chunks_sent += 1
                        yield _format_sse(data)

                    agent_completed = True

                except (asyncio.CancelledError, GeneratorExit):
                    client_disconnected = True
                    if agent_completed:
                        logging.debug("Client closed /examples connection after completion (chunks=%d)", chunks_sent)
                    else:
                        logging.warning("Client disconnected mid-stream on /examples after %d chunks", chunks_sent)
                    raise
                except Exception as e:
                    logging.error("Error while streaming examples output", exc_info=True)
                    yield _format_sse(json.dumps({"event": "error", "error": str(e)}, ensure_ascii=False))
                finally:
                    # make sure done always comes (unless the client is already gone)
                    if not client_disconnected:
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

                assistant_buffer: List[str] = []
                chunks_sent = 0
                agent_completed = False
                client_disconnected = False

                try:
                    async for kind, item in _with_heartbeat(
                        agent_fn(query_settings, server_settings, vector_store)
                    ):
                        if kind == "heartbeat":
                            yield SSE_HEARTBEAT
                            continue

                        chunk = item
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

                        chunks_sent += 1
                        yield _format_sse(data)

                    # Agent generator finished producing all output normally.
                    agent_completed = True

                except (asyncio.CancelledError, GeneratorExit):
                    client_disconnected = True
                    if agent_completed:
                        # Connection dropped after the answer was fully produced —
                        # benign tail-end close (e.g. client released the reader).
                        logging.debug(
                            "Client closed connection after stream completed (session=%s, chunks=%d)",
                            session_id, chunks_sent,
                        )
                    else:
                        # Connection dropped while the agent was still producing —
                        # the answer was genuinely cut off (proxy timeout, app
                        # backgrounded, network switch, navigation, etc.).
                        logging.warning(
                            "Client disconnected mid-stream: answer cut off after %d chunks (session=%s)",
                            chunks_sent, session_id,
                        )
                    raise
                except Exception as e:
                    logging.error("Error while streaming agent output", exc_info=True)
                    yield _format_sse(json.dumps({"event": "error", "error": str(e)}, ensure_ascii=False))
                finally:
                    # 5) store assistant full answer ONCE (even on a partial/cut-off stream)
                    full_answer = "".join(assistant_buffer).strip()
                    if session_id:
                        # hent siste historikk igjen i tilfelle andre forespørsler har skrevet
                        final_history = _load_history(session_id)
                        _store_assistant_message(session_id, final_history, full_answer)

                    # Can't (and needn't) emit a final frame once the client is gone —
                    # yielding during GeneratorExit raises, and the socket is closed.
                    if not client_disconnected:
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