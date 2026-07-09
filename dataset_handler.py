from datasets import load_dataset
dataset=load_dataset("theatticusproject/cuad-qa",trust_remote_code=True)
example=dataset["train"][0]
with open("data/processed/sample_contract.txt","w",encoding="utf-8") as f:
    f.write(example["context"])
