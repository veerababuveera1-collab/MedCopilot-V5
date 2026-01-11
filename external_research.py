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

        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",   # ✅ Supported model
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
        return {
            "answer": f"❌ External AI Error: {str(e)}"
        }
