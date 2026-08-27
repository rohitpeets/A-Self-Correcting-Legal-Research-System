
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
_client=Groq(api_key=os.getenv("GROQ_API_KEY"))

VALID_ROUTES={"keyword","semantic","hybrid"}
DEFAULT_ROUTE="hybrid"

SYSTEM_PROMPT="""You are a query router for a legal contract search system.
Every question you receive is about the text of a commercial contract.
Classify the question into exactly one category, based on how the question's
wording relates to the contract's wording.
keyword — The question contains literal strings that will appear verbatim in
the contract: quoted phrases, capitalized defined terms, section or clause
numbers, dollar amounts, dates, or named parties. Exact word matching will
find the right passage.
Example: What does Section 8.2 say about "Force Majeure"?

semantic — The question is phrased in everyday language that will NOT appear
in the contract. The contract expresses the same idea in formal legal wording.
Meaning has to be matched, because the words won't match.
Example: Can the tenant get out of the lease early?
    
hybrid — The question mixes both: an everyday-language concept AND at least
one literal term likely to appear verbatim in the contract.
Example: How much notice must the Lessee give before terminating?
    
Respond with exactly one word: keyword, semantic, or hybrid."""

def parse_route(raw):
    if not raw:
        return DEFAULT_ROUTE
    cleaned=raw.strip().lower().strip(".,!?:;'\"")
    if cleaned in VALID_ROUTES:
        return cleaned
    for token in cleaned.split():
        token=token.strip(".,!?:;'\"")
        if token in VALID_ROUTES:
            return token
    return DEFAULT_ROUTE
            
def classify(query):
    """Return one of: 'keyword', 'semantic', 'hybrid'. Falls back on failure."""
    try:
        response = _client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": query},
            ],
            temperature=0,
            max_tokens=512,
        )
        answer=response.choices[0].message.content
        print("RAW:",repr(answer))
        return parse_route(answer)
    except Exception as e:
        print("CLASSIFIER FAILED:", repr(e))
        return DEFAULT_ROUTE
