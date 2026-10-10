from flask import Flask, render_template, jsonify, request
from ai_services.servam_api import generate_answer
from utils.pdf_to_read import pdf_to_text_extract
from utils.chroma_db_config import (
    query_vector_store,
    get_collection_stats,
    get_or_create_collection
)
from utils.db import (
    init_db,
    save_conversation,
    get_last_n_qa_pairs,
    get_session_history,
    clear_session_history
)
import time
from ingest import ingest_documents
from utils.logger import get_logger

logger = get_logger("app")


app = Flask(__name__)
def init_knowledge_base():
    """
    Checks if ChromaDB vector store contains indexed chunks.
    If empty, automatically triggers PDF ingestion from the data/ directory.
    """
    try:
        col = get_or_create_collection()
        stats = get_collection_stats(col)
        if stats.get("count", 0) == 0:
            print("[Startup] Chroma collection is empty. Performing initial ingestion from 'data/'...")
            logger.info("Chroma collection is empty. Performing initial ingestion from 'data/'...")
            ingest_documents(data_dir="data")
        else:
           logger.info("Chroma knowledge base ready with %d chunk(s).", stats['count'])
    except Exception as e:
        logger.exception("Error verifying/initializing ChromaDB knowledge base: %s", e)

# Initialize collection on application load
init_knowledge_base()

# Initialize SQLite database for conversation history
try:
    init_db()
    logger.info("SQLite conversation history database initialized.")
except Exception as e:
    logger.exception("Error initializing SQLite database: %s", e)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_response', methods=['POST'])
def get_response():
    start_time = time.time()
    user_data = request.get_json(silent=True) or {}
    user_message = user_data.get("message", "").strip()
    session_id = user_data.get("session_id", "").strip() or "default"

    logger.info(
        "Incoming /get_response [session_id='%s', ip='%s', message_len=%d]",
        session_id, request.remote_addr, len(user_message)
    )

    if not user_message:
        logger.warning("Empty or blank user message received for session '%s'", session_id)
        return jsonify({
            "reply": "Please provide a valid question or topic.",
            "sources": [],
            "session_id": session_id
        })
    try:
        # Retrieve the last 5 Q&A pairs for conversation context
        history_pairs = get_last_n_qa_pairs(session_id, limit=5)

        # Retrieve top relevant context chunks from ChromaDB PersistentClient
        chunks = query_vector_store(user_message, n_results=3)

        # Format deduplicated sources for the UI
        sources = []
        seen_sources = set()

        for chunk in chunks:
         meta = chunk.get("metadata", {})
        source_name = meta.get("source", "Document")
        page_num = meta.get("page")
        label = f"{source_name} (Page {page_num})" if page_num else source_name
        if label not in seen_sources:
            seen_sources.add(label)
            sources.append(label)

        logger.debug(
            "Session '%s': Retrieved %d history pair(s), %d chunk(s), %d source(s)",
            session_id, len(history_pairs), len(chunks), len(sources)
        )

     # Generate answer grounded in the retrieved chunks and conversation history
        answer = generate_answer(user_message, chunks, history=history_pairs)

    # Save conversation into SQLite
        try:
            save_conversation(session_id, user_message, answer, sources)
        except Exception as e:
            logger.exception("Session '%s': Error saving conversation history to SQLite: %s", session_id, e)

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "/get_response completed in %d ms [session_id='%s', reply_len=%d, sources_count=%d]",
            elapsed_ms, session_id, len(answer), len(sources)
        )


        return jsonify({
        "reply": answer,
        "sources": sources
    })
    except Exception as e:
        logger.exception("Unexpected error processing /get_response for session '%s': %s", session_id, e)
        return jsonify({
            "reply": "An unexpected error occurred while processing your request.",
            "sources": [],
            "session_id": session_id
        }), 500
    
@app.route("/api/history", methods=["GET"])
def get_history():
    """
    Returns the message history for a given session.
    """
    session_id = request.args.get("session_id", "default").strip() or "default"
    logger.info("GET /api/history for session '%s'", session_id)
    try:
        history = get_session_history(session_id)
        return jsonify({"status": "success", "session_id": session_id, "history": history})
    except Exception as e:
        logger.exception("Error in GET /api/history for session '%s': %s", session_id, e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/history/clear", methods=["POST"])
@app.route("/clear_chat", methods=["POST"])
def clear_history():
    """
    Clears conversation history for a given session.
    """
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "default").strip() or "default"
    logger.info("POST /clear_chat for session '%s'", session_id)
    try:
        clear_session_history(session_id)
        return jsonify({"status": "success", "message": f"History cleared for session: {session_id}"})
    except Exception as e:
        logger.exception("Error in POST /clear_chat for session '%s': %s", session_id, e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/new_chat", methods=["POST"])
def new_chat():
    """
    Acknowledge a new chat session.
    """
    data = request.get_json(silent=True) or {}
    new_sid = data.get("session_id", "unknown")
    old_sid = data.get("old_session_id", "unknown")
    logger.info("POST /new_chat initiated [old_sid='%s' -> new_sid='%s']", old_sid, new_sid)
    return jsonify({"status": "success", "message": "New chat session started."})

@app.route("/api/kb/stats", methods=["GET"])
def get_kb_stats():
    """
    Returns statistics about the current ChromaDB vector store.
    """
    try:
        stats = get_collection_stats()
        logger.info("GET /api/kb/stats: %s", stats)
        return jsonify({"status": "success", "stats": stats})
    except Exception as e:
        logger.exception("Error in GET /api/kb/stats: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/kb/reindex", methods=["POST"])
def reindex_kb():
    """
    Triggers re-indexing of all PDFs in the data folder.
    """
    logger.info("POST /api/kb/reindex triggered")
    try:
        count = ingest_documents(data_dir="data", reset=True)
        logger.info("Reindex completed: %d chunk(s) indexed", count)
        return jsonify({
            "status": "success",
            "message": f"Successfully reindexed {count} chunk(s)."
        })
    except Exception as e:
        logger.exception("Error in POST /api/kb/reindex: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)