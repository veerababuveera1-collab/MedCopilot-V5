import os
from groq import Groq

# Safe limits for Groq free tier
MAX_PROMPT_CHARS = 4500
MAX_RESPONSE_TOKENS = 1024


def trim_prompt(text, max_chars=MAX_PROMPT_CHARS):
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def external_research_answer(query):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return {
            "answer": "❌ External AI not configured. Please set GROQ_API_KEY in Streamlit Secrets."
        }

    # Protect Groq from oversized prompts
    safe_query = trim_prompt(query)

    try:
        client = Groq(api_key=api_key)

        # Model fallback order (best → fastest)
        models = [
            "llama-3.3-70b-versatile",
            "llama-3.2-90b-text-preview",
            "llama-3.1-8b-instant"
        ]

        last_error = None

        for model in models:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a senior medical research and clinical decision support AI. Provide accurate, evidence-based medical responses."
                        },
                        {
                            "role": "user",
                            "content": safe_query
                        }
                    ],
                    temperature=0.2,
                    max_tokens=MAX_RESPONSE_TOKENS
                )

                return {
                    "answer": response.choices[0].message.content
                }

            except Exception as e:
                last_error = str(e)
                continue

        return {
            "answer": f"❌ All Groq models failed. Last error: {last_error}"
        }

    except Exception as e:
        return {
            "answer": f"❌ External AI Error: {str(e)}"
        }
