from app import chunks

term="governing law"
found=0
for i in range(len(chunks)):
    if term in chunks[i].lower():
        found=found+1
        print(f"chunk_{i}:")
        print(chunks[i])
        print()
print(f"chunks containing '{term}': {found}")