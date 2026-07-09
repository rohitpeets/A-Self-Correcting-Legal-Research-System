
import re
import  nltk
nltk.download('punkt_tab')
from dataset_handler import example
test_case=example["context"]
def chunk_contract(input:str,no_lines:int=5,overlap:int=1)->list[str]:
    cleaned_txt=re.sub(r'\s+',' ',input)
    sentences=nltk.sent_tokenize(cleaned_txt)
    no_of_sentences=len(sentences)
    stack=[]
    chunks=[]

    start=0
    while(start<no_of_sentences):
        for i in range (start,min(start + no_lines, no_of_sentences)):
            stack.append(sentences[i])
        chunks.append(" ".join(stack))
        stack.clear()
        start=start+no_lines-overlap
    return chunks
chunks = chunk_contract(test_case)
print("Total sentences:", len(nltk.sent_tokenize(test_case)))
print("Number of chunks:", len(chunks))
print("--- Chunk 1 ---")
print(chunks[0])
print("--- Chunk 2 ---")
print(chunks[1])
    


            







