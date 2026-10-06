from sarvamai import SarvamAI
from dotenv import load_dotenv
import os
from typing import Union, List, Dict, Any

load_dotenv()


def generate_answer(user_message: str, context: Union[str, List[Dict[str, Any]]]) -> str:
    """
    Generates a grounded response to the user's message using retrieved context from ChromaDB.
    """
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        return "Sarvam AI API key is not configured. Please set SARVAM_API_KEY in your .env file."

    client = SarvamAI(api_subscription_key=api_key)

    # Format context if passed as retrieved chunks
    if isinstance(context, list):
        if not context:
            return "I couldn't find any relevant admission or syllabus information in the knowledge base for your query."

        formatted_chunks = []
        for i, chunk in enumerate(context):
            meta = chunk.get("metadata", {})
            source = meta.get("source", "Document")
            page = meta.get("page", "?")
            text = chunk.get("text", "")
            formatted_chunks.append(f"[Source: {source} | Page: {page}]\n{text}")
        context_str = "\n\n".join(formatted_chunks)
    else:
        context_str = str(context)

    prompt = f"""You have to generate answer strcitly from the given text {context_str} only.
      If you don't find answer in the given text say answer not found. don't give your own answer. 
      user question: {user_message}

"""

    model_name = os.getenv("SARVAM_MODEL", "sarvam-105b")

    try:
        response = client.chat.completions(
            model=model_name,
            messages=[
                {"role": "user",
                 "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content

    except Exception as e:
<<<<<<< HEAD
        return f"An error occurred while generating the answer: {str(e)}"
=======
        return f"An error occurred while generating the answer: {str(e)}"
>>>>>>> 1671daa284be8d5f44e7f97c31b551b0711cc43c
