import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def external_research_answer(query):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a medical research assistant."},
            {"role": "user", "content": query}
        ],
        temperature=0.2
    )

    return {
        "answer": response.choices[0].message.content
    }
