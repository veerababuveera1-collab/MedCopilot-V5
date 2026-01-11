import os
from groq import Groq

def external_research_answer(query):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return {
            "answer": "❌ External AI not configured. Please set GROQ_API_KEY in Streamlit Secrets."
        }

    try:
        client = Groq(api_key=api_key)

        # Try latest supported models in order
        models = [
            "llama-3.3-70b-versatile",   # newest
            "llama-3.2-90b-text-preview",
            "llama-3.1-8b-instant"
        ]

        last_error = None

        for model in models:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a medical research and clinical decision support assistant."},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.2
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
