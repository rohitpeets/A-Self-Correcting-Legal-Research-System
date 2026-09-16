import os
from dotenv import load_dotenv
from groq import Groq

from settings import MAX_QUERY_CHARS, GUARD_MODEL, GUARD_INJECTION_THRESHOLD

load_dotenv()
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def check_length(query):
    """Return True if the query is within the allowed length."""
    return len(query) <= MAX_QUERY_CHARS


def check_injection(query):
    """Return True if the query looks safe, False if it looks like a prompt-injection/jailbreak attempt.

    Uses Groq's Prompt Guard model, which returns the injection probability
    as a bare numeric string in place of a normal chat reply. Fails closed:
    any error from the guard model is treated as unsafe.
    """
    try:
        response = _client.chat.completions.create(
            model=GUARD_MODEL,
            messages=[{"role": "user", "content": query}],
        )
        score = float(response.choices[0].message.content)
    except Exception as e:
        print("GUARDRAIL CHECK FAILED:", repr(e))
        return False
    return score < GUARD_INJECTION_THRESHOLD


def check_query(query):
    """Run all input guardrails on 'query'.

    Returns (True, None) if the query is safe to process, or
    (False, reason) if a guardrail tripped.
    """
    if not check_length(query):
        return False, "query too long"
    if not check_injection(query):
        return False, "prompt injection detected"
    return True, None
