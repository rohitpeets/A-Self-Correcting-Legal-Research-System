import os
from dotenv import load_dotenv
from groq import Groq

from settings import VERIFY_MODEL, VERIFY_TEMPERATURE

load_dotenv()
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a grounding checker for a legal research assistant.
You are given numbered passages from a contract and an answer that was generated
from those passages, with citations like [1].

Check whether every factual claim in the answer is actually supported by the
cited passages. The answer fails the check if it states anything not present in
the passages, adds outside legal knowledge, or cites a passage that doesn't
support the claim next to it.

Respond with exactly one word: YES if the answer is fully grounded, NO if it is not."""

def verify(query, answer, context):
    """Return True if 'answer' is grounded in 'context', False otherwise.

    Fails closed: any error from the judge model is treated as ungrounded.
    """
    user_prompt = f"Passages:\n\n{context}\n\nQuestion: {query}\n\nAnswer:\n{answer}"

    try:
        response = _client.chat.completions.create(
            model=VERIFY_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=VERIFY_TEMPERATURE,
            max_tokens=512,
        )
        verdict = (response.choices[0].message.content or "").strip().upper()
    except Exception as e:
        print("VERIFICATION FAILED:", repr(e))
        return False

    return verdict.startswith("YES")
