"""
Shared streaming chat response helper.

Both the enrolled-student route (`student.chat_stream`) and the public/admin
slug route (`app.chat_stream_api`) delegate to `stream_chat_response()` here.
This is the single source of truth for:
  - Session resolution (create / reuse / retroactive user binding)
  - User-message persistence (text + optional image)
  - Conversation history loading and formatting (with multi-turn image continuity)
  - Retrieval invocation and context assembly
  - Hybrid-mode + limited-context detection
  - LLM streaming + SSE event emission (multimodal-aware)
  - Assistant-message persistence
  - QA logging

Path-specific concerns (route definitions, auth decorators, TA-lookup-by-slug,
email verification gates) remain in the calling files where they belong.
"""

import base64
import json
import logging
import secrets
import time
import traceback
from datetime import datetime

from flask import Response, stream_with_context

from config import Config

logger = logging.getLogger(__name__)

# Per-session cap on student image uploads — prevents cost runaway from a single user.
# Counts user messages with image_data on the same chat session.
MAX_IMAGES_PER_SESSION = 20

# Per-upload size and format constraints (server-side; client also enforces).
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/heic", "image/heif"}


def parse_chat_request(request):
    """
    Pull (query, session_id, image_data, image_mime, error) from a chat-stream HTTP request.
    Handles both JSON body (legacy text-only) and multipart/form-data (with optional image).

    Returns a tuple. If `error` is non-None, callers should return a 400 with that message.
    """
    content_type = (request.content_type or "").lower()
    is_multipart = content_type.startswith("multipart/")

    if is_multipart:
        query = (request.form.get("query") or "").strip()
        session_id = request.form.get("session_id") or ""
        image_file = request.files.get("image")
        if image_file and image_file.filename:
            mime = (image_file.mimetype or "").lower()
            if mime not in ALLOWED_IMAGE_MIMES:
                return query, session_id, None, None, f"Unsupported image format: {mime or 'unknown'}"
            data = image_file.read()
            if len(data) > MAX_IMAGE_BYTES:
                return query, session_id, None, None, "Image exceeds 5 MB limit"
            if not data:
                return query, session_id, None, None, "Empty image file"
            return query, session_id, data, mime, None
        return query, session_id, None, None, None

    body = request.json or {}
    return (body.get("query") or "").strip(), body.get("session_id") or "", None, None, None


def _build_history_for_llm(messages, max_turns: int = 6) -> list:
    """
    Build a structured conversation-history list for the LLM, preserving images.

    Returns a list of OpenAI-compatible message dicts where any user message with
    an attached image becomes a multimodal content list. Assistant messages stay
    as plain strings.
    """
    if not messages:
        return []

    out = []
    for msg in messages[-(max_turns):]:
        if msg.role == "user" and getattr(msg, "image_data", None):
            mime = msg.image_mime or "image/jpeg"
            b64 = base64.b64encode(msg.image_data).decode("ascii")
            text_part = (msg.content or "")[:600]
            out.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": text_part},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            })
        else:
            text = (msg.content or "")[:600]
            out.append({"role": msg.role, "content": text})
    return out


def stream_chat_response(
    *,
    ta,
    query: str,
    session_id: str,
    user_id,
    is_anonymous: bool,
    image_data: bytes = None,
    image_mime: str = None,
) -> Response:
    """
    Run the full chat streaming pipeline for a single user query.

    Args:
        ta: TeachingAssistant ORM object (already validated by caller — must be
            indexed and accessible to this user)
        query: the student's question (already trimmed and non-empty)
        session_id: existing session id, or empty string to create a new session
        user_id: int (student or admin) or None (truly anonymous)
        is_anonymous: True for slug-based public route, False for enrolled-student route.
            Drives only the QA-log flag — does not change retrieval behavior.
        image_data: optional raw image bytes co-submitted with the text query.
            Only honored when ta.image_upload_enabled is True. Subject to per-session cap.
        image_mime: MIME type for the image (e.g. 'image/png', 'image/jpeg'). Required when
            image_data is provided.

    Returns:
        Flask streaming Response (SSE: text/event-stream)
    """
    from models import db, ChatSession, ChatMessage

    # IMAGE GUARD: reject silently-attached images for TAs that don't have the feature.
    # Frontend should hide the upload control, but this is the defense-in-depth check.
    if image_data and not ta.image_upload_enabled:
        logger.warning(
            f"[{ta.id}] Image upload attempted but feature not enabled — dropping image bytes"
        )
        image_data = None
        image_mime = None

    # SESSION RESOLUTION
    # Either reuse the supplied session (with security check + retroactive admin binding),
    # or create a fresh one. Always commit before launching the generator so the user
    # message has a stable session_id.
    if not session_id:
        session_id = secrets.token_urlsafe(16)
        chat_session = ChatSession(id=session_id, ta_id=ta.id, user_id=user_id)
        db.session.add(chat_session)
        db.session.commit()
    else:
        chat_session = ChatSession.query.filter_by(id=session_id, ta_id=ta.id).first()
        if not chat_session:
            session_id = secrets.token_urlsafe(16)
            chat_session = ChatSession(id=session_id, ta_id=ta.id, user_id=user_id)
            db.session.add(chat_session)
            db.session.commit()
        elif user_id is not None and not chat_session.user_id:
            # Retroactive binding: an authenticated user (admin or student) is now
            # using a session that started anonymous. Stamp the user_id.
            chat_session.user_id = user_id
            db.session.commit()

    # PER-SESSION IMAGE CAP: enforce before persisting so we don't half-commit.
    # Returns a streaming-style error response so the UX matches a normal chat error.
    if image_data:
        existing_image_count = (
            ChatMessage.query
            .filter_by(session_id=session_id, role="user")
            .filter(ChatMessage.image_data.isnot(None))
            .count()
        )
        if existing_image_count >= MAX_IMAGES_PER_SESSION:
            logger.info(
                f"[{ta.id}] Session {session_id} hit image cap "
                f"({existing_image_count}/{MAX_IMAGES_PER_SESSION}) — rejecting upload"
            )
            def cap_error():
                yield (
                    "data: " + json.dumps({
                        "type": "error",
                        "message": (
                            f"You've uploaded the maximum of {MAX_IMAGES_PER_SESSION} images "
                            "in this session. Start a new conversation to upload more."
                        ),
                    }) + "\n\n"
                )
            return Response(
                stream_with_context(cap_error()),
                mimetype='text/event-stream',
                headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
            )

    # USER MESSAGE PERSISTENCE — text + optional image, atomic single row
    user_message = ChatMessage(
        session_id=session_id,
        role="user",
        content=query,
        image_data=image_data,
        image_mime=image_mime,
    )
    db.session.add(user_message)
    db.session.commit()

    # Capture TA fields BEFORE the generator. Flask streaming generators run after
    # the request context is gone, so we can't lazy-access ORM relations there.
    ta_id = ta.id
    ta_slug = ta.slug
    ta_name = ta.name
    ta_system_prompt = ta.system_prompt
    ta_course_name = ta.course_name

    def generate():
        # Imports inside the generator so failures don't prevent the SSE response from
        # opening; we'd rather stream an 'error' event than 500.
        from src.retriever import retrieve_context
        from src.response_generator import generate_response_stream, escape_hash_in_latex
        from src.qa_logger import log_qa_entry

        start_time = time.time()
        retrieval_latency_ms = 0
        generation_latency_ms = 0
        chunk_count = 0
        sources = []
        full_response = ""

        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching course materials...'})}\n\n"

            # CONVERSATION HISTORY
            recent_messages = (
                ChatMessage.query
                .filter_by(session_id=session_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(10)
                .all()
            )
            conversation_history = list(reversed(recent_messages))

            history_text = ""
            if conversation_history:
                history_parts = []
                for msg in conversation_history[-6:]:
                    role_label = "Student" if msg.role == "user" else "Assistant"
                    history_parts.append(f"{role_label}: {msg.content[:300]}...")
                history_text = "\n".join(history_parts)

            # Build a structured history list for the LLM. Used when ANY message
            # in the recent window has an image, so multi-turn follow-ups ("what
            # about the slope here?") can still see the prior drawing. When no
            # images are involved, history_for_llm stays None and the LLM call
            # uses the existing prose history_text.
            session_has_any_image = any(
                getattr(m, "image_data", None) for m in conversation_history
            ) or bool(image_data)
            history_for_llm = (
                _build_history_for_llm(conversation_history, max_turns=6)
                if session_has_any_image
                else None
            )

            # RETRIEVAL
            retrieval_start = time.time()
            chunks, retrieval_diagnostics = retrieve_context(
                ta_id, query, top_k=8,
                conversation_history=conversation_history,
                session_id=session_id,
            )
            retrieval_latency_ms = int((time.time() - retrieval_start) * 1000)
            chunk_count = len(chunks)

            yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing relevant content...'})}\n\n"

            # CONTEXT ASSEMBLY: primary chunks first, teaching material clearly tagged after.
            primary = [c for c in chunks if c.get('retrieval_role') != 'teaching_material']
            teaching = [c for c in chunks if c.get('retrieval_role') == 'teaching_material']
            parts = [f"[From: {c['file_name']}]\n{c['text']}" for c in primary]
            if teaching:
                parts.append("[RELEVANT TEACHING MATERIAL FROM COURSE LECTURES]")
                parts.extend(f"[From: {c['file_name']}]\n{c['text']}" for c in teaching)
            context = "\n\n---\n\n".join(parts)

            sources = list(dict.fromkeys(c['file_name'] for c in chunks[:8]))[:3]
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"

            # HYBRID + LIMITED CONTEXT
            hybrid_mode = retrieval_diagnostics.get("hybrid_fallback_triggered", False)
            hybrid_doc_filename = retrieval_diagnostics.get("hybrid_doc_filename")
            query_reference = retrieval_diagnostics.get("validation_expected_ref")
            attempt_count = retrieval_diagnostics.get("attempt_count", 0)

            score_top1 = retrieval_diagnostics.get("score_top1", 0) or 0
            total_chunks_in_ta = retrieval_diagnostics.get("total_chunks_in_ta", 0) or 0
            supplementary_found = retrieval_diagnostics.get("supplementary_teaching_found", False)
            limited_context = (
                chunk_count == 0
                or (chunk_count <= 2 and score_top1 < 0.5)
                or (hybrid_mode and score_top1 < 0.6)
                or total_chunks_in_ta <= 5
            ) and not supplementary_found

            # LLM STREAMING (multimodal-aware)
            current_image = None
            if image_data:
                current_image = {"data": image_data, "mime": image_mime or "image/jpeg"}

            generation_start = time.time()
            for token in generate_response_stream(
                query=query,
                context=context,
                system_prompt=ta_system_prompt,
                conversation_history=history_text,
                course_name=ta_course_name,
                hybrid_mode=hybrid_mode,
                hybrid_doc_filename=hybrid_doc_filename,
                query_reference=query_reference,
                attempt_count=attempt_count,
                limited_context=limited_context,
                current_image=current_image,
                history_for_llm=history_for_llm,
            ):
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            generation_latency_ms = int((time.time() - generation_start) * 1000)

            full_response = escape_hash_in_latex(full_response)

            # ASSISTANT MESSAGE PERSISTENCE
            chat_session_update = ChatSession.query.get(session_id)
            assistant_message = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=full_response,
                sources=sources,
            )
            db.session.add(assistant_message)
            if chat_session_update:
                chat_session_update.last_activity = datetime.utcnow()
            db.session.commit()

            # QA LOG
            total_latency_ms = int((time.time() - start_time) * 1000)
            token_count = len(full_response.split())

            log_qa_entry(
                ta_id=str(ta_id),
                ta_slug=ta_slug,
                ta_name=ta_name,
                course_name=ta_course_name,
                session_id=session_id,
                query=query,
                answer=full_response,
                sources=sources,
                chunk_count=chunk_count,
                latency_ms=total_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                token_count=token_count,
                retrieval_diagnostics=retrieval_diagnostics,
                llm_model=Config.LLM_MODEL,
                is_anonymous=is_anonymous,
            )

            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"

        except Exception as e:
            logger.error(f"Streaming chat error for TA {ta_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'message': 'An error occurred processing your question.'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )
