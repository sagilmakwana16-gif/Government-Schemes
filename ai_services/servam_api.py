import os
import time
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
from sarvamai import SarvamAI
from utils.logger import get_logger

logger = get_logger("servam_api.py")

load_dotenv()


def generate_answer(
    user_message: str,
    context: Union[str, List[Dict[str, Any]]],
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Generates a grounded response to the user's message using retrieved context from ChromaDB

    and previous conversation history.
    """
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        logger.error("Sarvam AI API key is not configured in environment.")
        return "Sarvam AI API key is not configured. Please set SARVAM_API_KEY in your .env file."

    client = SarvamAI(api_subscription_key=api_key)

    # Format context if passed as retrieved chunks
    if isinstance(context, list):
        if not context and not history:
            logger.info("No context and no history provided for user query.")
            return "I couldn't find any relevant admission or syllabus information in the knowledge base for your query."

        formatted_chunks = []
        for chunk in context:
            meta = chunk.get("metadata", {})
            source = meta.get("source", "Document")
            page = meta.get("page", "?")
            text = chunk.get("text", "")
            formatted_chunks.append(f"\n{text}")

        context_str = (
            "\n\n".join(formatted_chunks)
            if formatted_chunks
            else "No specific document context found."
        )
    else:
        context_str = str(context)

    # Format recent conversation history
    history_section = ""
    history_count = 0
    if history:
        formatted_history = []
        for pair in history:
            q = pair.get("question", "").strip()
            a = pair.get("answer", "").strip()
            if q or a:
                formatted_history.append(f"User: {q}\nAssistant: {a}")
        if formatted_history:
            history_count = len(formatted_history)
            history_section = (
                "Recent Conversation History:\n"
                + "\n\n".join(formatted_history)
                + "\n\n"
            )

    prompt = f"""You are a helpful Government Scheme Assistant. 

Guidelines:
1. For general greetings (e.g., "hi", "hello", "my name is..."), respond politely and ask how you can help them with government schemes.
2. For specific questions about schemes, base your answer strictly on the Context below. Do not make up facts.
3. If the user asks a question about schemes/rules that is NOT in the Context, clearly state: "I couldn't find information about this in the available documents."

{history_section}Context:

{history_section}Context:
{context_str}

User Question:
{user_message}
"""

    model_name = os.getenv("SARVAM_MODEL", "sarvam-105b")
    logger.info(
        "Requesting answer using model '%s' (history_turns=%d, context_len=%d chars)",
        model_name,
        history_count,
        len(context_str),
    )

    start_time = time.time()

    try:
        response = client.chat.completions(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        end_time = time.time()
        elapsed_ms = int((end_time - start_time) * 1000)

        # Safety check: extract content safely from response object
        output_text = ""
        if response and getattr(response, "choices", None) and len(response.choices) > 0:
            output_text = response.choices[0].message.content or ""
        elif hasattr(response, "output_text"):
            output_text = response.output_text or ""

        if not output_text:
            output_text = "I couldn't generate a response based on the provided documents."

        logger.info(
            "Sarvam response generated successfully in %d ms (%d chars)",
            elapsed_ms,
            len(output_text),
        )
        return output_text

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error(
            "Primary generation using '%s' failed after %d ms. Error: %s",
            model_name,
            elapsed_ms,
            str(e),
        )
        return f"An error occurred while generating the answer: {str(e)}"