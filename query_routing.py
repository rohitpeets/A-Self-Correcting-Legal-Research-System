from config import query
import re
from dotenv import load_dotenv
import os
from groq import Groq

text='what is "Rohitpreet singh"'


def quote_test(query):
    key=re.search(r'"[^"]+"', query)
    if(key!=None):
        return True
    else:
        return False

def defination_test(input_query):
    words=["what is","define"]
    out=False
    for word in words:
        if(word in input_query.lower()):
            out=True
    return out

def section_test(input_query):
    pattern = r'(section|clause)\s+\d+(\.\d+)?'
    key = re.search(pattern, input_query, re.IGNORECASE)
    if key != None:
        return True
    else:
        return False

    
def is_keyword_query(in_text):
    if(quote_test(in_text) or defination_test(in_text) or section_test(in_text)):
        return True
    else:
        return False

def llm_output_parser(s):
    list=s.split(" ")
    targets=["semantic","hybrid","keyword"]
    for word in list:
        for i in targets:
            if(i == word):
                return i

            
def llm_classifier_and_run(query):
    load_dotenv()
    client=Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt=(query+"This is a user given query.You are to carry out the following:"+
        "Classify the given query into three categories as follows:"+
        "1)semantic: if a dense search only approach would be more ideal than either bm25 keyword dearch or dense_bm25 search+rrf"+
        "2)keyword: if bm25 search only would be more ideal than either dense search or dense+bm25 search+rrf "+
        "3)Hybrid: if both dense+bm25 search+rrf is the ideal approach "+
        "Respond with ONLY the single word: semantic, keyword, or hybrid. No explanation, no punctuation, nothing else.")
    response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content":prompt}
    ]
)
    answer = response.choices[0].message.content
    print("RAW LLM ANSWER:", answer)
    route_var=llm_output_parser(answer)
    return route_var


def query_router(query):
    if(is_keyword_query(query)):
        return "keyword"
    else:
        return llm_classifier_and_run(query)
print(query_router('what is "Force Majeure"'))       # should print "keyword" (quotes)
print(query_router("can the tenant leave early?"))    # should hit the LLM fallback