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
# Counts ChatMessageImage rows on the same chat session.
MAX_IMAGES_PER_SESSION = 20
# Per-turn cap — limits a single message to a sane number of attached images.
MAX_IMAGES_PER_TURN = 5

# Per-upload size and format constraints (server-side; client also enforces).
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB per image
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/heic", "image/heif"}


def parse_chat_request(request):
    """
    Pull (query, session_id, images, error) from a chat-stream HTTP request.

    Handles both JSON body (legacy text-only) and multipart/form-data with one
    or more `image` file parts. Multiple files preserve client-side upload order.

    `images` is a list of {"data": bytes, "mime": str} dicts — empty if no images.
    On any validation error, `error` is non-None and callers should return 400.
    """
    content_type = (request.content_type or "").lower()
    is_multipart = content_type.startswith("multipart/")

    if is_multipart:
        query = (request.form.get("query") or "").strip()
        session_id = request.form.get("session_id") or ""
        image_files = request.files.getlist("image")
        # Filter out empty file slots (browsers sometimes submit empty FormData entries)
        image_files = [f for f in image_files if f and f.filename]

        if len(image_files) > MAX_IMAGES_PER_TURN:
            return query, session_id, [], f"Too many images — max {MAX_IMAGES_PER_TURN} per message"

        images = []
        for f in image_files:
            mime = (f.mimetype or "").lower()
            if mime not in ALLOWED_IMAGE_MIMES:
                return query, session_id, [], f"Unsupported image format: {mime or 'unknown'}"
            data = f.read()
            if len(data) > MAX_IMAGE_BYTES:
                return query, session_id, [], f"Image '{f.filename}' exceeds 5 MB limit"
            if not data:
                return query, session_id, [], f"Empty image file: {f.filename}"
            images.append({"data": data, "mime": mime})
        return query, session_id, images, None

    body = request.json or {}
    return (body.get("query") or "").strip(), body.get("session_id") or "", [], None


def _build_history_for_llm(messages, max_turns: int = 6) -> list:
    """
    Build a structured conversation-history list for the LLM, preserving images.

    Returns a list of OpenAI-compatible message dicts where any user message with
    one or more attached images becomes a multimodal content list with the text
    part followed by image parts in display order. Assistant messages stay as
    plain strings.
    """
    if not messages:
        return []

    out = []
    for msg in messages[-(max_turns):]:
        attached = list(msg.images) if (msg.role == "user" and getattr(msg, "images", None)) else []
        if attached:
            text_part = (msg.content or "")[:600]
            content = [{"type": "text", "text": text_part}]
            for img in attached:
                mime = img.image_mime or "image/jpeg"
                b64 = base64.b64encode(img.image_data).decode("ascii")
                content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
            out.append({"role": "user", "content": content})
        else:
            out.append({"role": msg.role, "content": (msg.content or "")[:600]})
    return out


def stream_chat_response(
    *,
    ta,
    query: str,
    session_id: str,
    user_id,
    is_anonymous: bool,
    images: list = None,
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
        images: optional list of {"data": bytes, "mime": str} dicts in display order.
            Only honored when ta.image_upload_enabled is True. Subject to MAX_IMAGES_PER_TURN
            and MAX_IMAGES_PER_SESSION caps.

    Returns:
        Flask streaming Response (SSE: text/event-stream)
    """
    from models import db, ChatSession, ChatMessage, ChatMessageImage

    images = images or []

    # IMAGE GUARD: reject silently-attached images for TAs that don't have the feature.
    # Frontend should hide the upload control, but this is the defense-in-depth check.
    if images and not ta.image_upload_enabled:
        logger.warning(
            f"[{ta.id}] Image upload attempted but feature not enabled — dropping {len(images)} image(s)"
        )
        images = []

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

    # PER-SESSION IMAGE CAP: count existing ChatMessageImage rows on this session,
    # plus the new ones, against the cap. Reject before persisting if over.
    if images:
        existing_image_count = (
            db.session.query(ChatMessageImage)
            .join(ChatMessage, ChatMessage.id == ChatMessageImage.message_id)
            .filter(ChatMessage.session_id == session_id)
            .count()
        )
        if existing_image_count + len(images) > MAX_IMAGES_PER_SESSION:
            logger.info(
                f"[{ta.id}] Session {session_id} would exceed image cap "
                f"({existing_image_count} + {len(images)} > {MAX_IMAGES_PER_SESSION})"
            )
            def cap_error():
                remaining = max(0, MAX_IMAGES_PER_SESSION - existing_image_count)
                yield (
                    "data: " + json.dumps({
                        "type": "error",
                        "message": (
                            f"You've reached the maximum of {MAX_IMAGES_PER_SESSION} images "
                            f"in this session ({remaining} remaining). Start a new conversation to upload more."
                        ),
                    }) + "\n\n"
                )
            return Response(
                stream_with_context(cap_error()),
                mimetype='text/event-stream',
                headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
            )

    # USER MESSAGE PERSISTENCE — text first, then attached images as ChatMessageImage rows
    user_message = ChatMessage(
        session_id=session_id,
        role="user",
        content=query,
    )
    db.session.add(user_message)
    db.session.flush()  # need the message id for FK on the image rows
    for idx, img in enumerate(images):
        db.session.add(ChatMessageImage(
            message_id=user_message.id,
            image_data=img["data"],
            image_mime=img.get("mime"),
            order=idx,
        ))
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
            # in the recent window has at least one attached image, so multi-turn
            # follow-ups ("what about the slope here?") can still see prior drawings.
            # When no images are involved at all, history_for_llm stays None and
            # the LLM call uses the existing prose history_text.
            session_has_any_image = any(
                bool(getattr(m, "images", None)) for m in conversation_history
            ) or bool(images)
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
                course_name=ta_course_name,
            )
            retrieval_latency_ms = int((time.time() - retrieval_start) * 1000)
            chunk_count = len(chunks)

            # ADVERSARIAL / OFF-TOPIC SHORT-CIRCUIT
            # The contextualizer flagged this query as off-topic (greeting, jailbreak,
            # roleplay attack, answer-extraction trick). Skip retrieval/context/generation
            # entirely and stream a brief canned redirect. Saves ~10-15s rerank + ~10-15s
            # gpt-5.2 generation per dismissed query.
            if retrieval_diagnostics.get("adversarial_short_circuit"):
                # Prefer the LLM-tailored redirect produced by the contextualizer (varies by
                # off-topic category — greeting / frustration / jailbreak / cheating). Falls
                # back to the generic canned line if the contextualizer didn't produce one.
                tailored = (retrieval_diagnostics.get("redirect_message") or "").strip()
                canned_fallback = (
                    f"Let's stay on track. I'm here to help with {ta_course_name} — "
                    "what would you like help with from the course materials?"
                )
                full_response = tailored or canned_fallback
                yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"

                # Persist assistant message
                chat_session_update = ChatSession.query.get(session_id)
                assistant_message = ChatMessage(
                    session_id=session_id,
                    role="assistant",
                    content=full_response,
                    sources=[],
                )
                db.session.add(assistant_message)
                if chat_session_update:
                    chat_session_update.last_activity = datetime.utcnow()
                db.session.commit()

                # Log the dismissed turn so we can measure detection rate + tune the prompt
                total_latency_ms = int((time.time() - start_time) * 1000)
                log_qa_entry(
                    ta_id=str(ta_id),
                    ta_slug=ta_slug,
                    ta_name=ta_name,
                    course_name=ta_course_name,
                    session_id=session_id,
                    query=query,
                    answer=full_response,
                    sources=[],
                    chunk_count=0,
                    latency_ms=total_latency_ms,
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=0,
                    token_count=len(full_response.split()),
                    retrieval_diagnostics=retrieval_diagnostics,
                    llm_model=Config.LLM_MODEL,
                    is_anonymous=is_anonymous,
                )

                yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
                return  # Skip the rest of the pipeline

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

            # LLM STREAMING (multimodal-aware) — pass the full list of current-turn images
            current_images = [
                {"data": img["data"], "mime": img.get("mime") or "image/jpeg"}
                for img in images
            ] if images else None

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
                current_images=current_images,
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
