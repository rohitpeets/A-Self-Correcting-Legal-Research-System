from query_classifier import classify

CASES=[
    ('What does the "Governing Law" section say?',"keyword"),
    ('Define "Confidential Information".',"keyword"),
    ("What does Section 6.8 say?","keyword"),
    ("How can either side end this agreement early?","semantic"),
    ("What happens if someone breaks the agreement?","semantic"),
    ("Is either side allowed to share private business details?","semantic"),
    ("How much notice must the Distributor give to terminate?","hybrid"),
    ("What are the payment obligations under Section 2.3?","hybrid"),
    ("Can the Company assign its rights to another party?","hybrid"),
    ('What happens to "Confidential Information" after the agreement ends?',"hybrid"),
]

passed=0
for i in range(len(CASES)):
    query=CASES[i][0]
    expected=CASES[i][1]
    got=classify(query)
    if got==expected:
        passed=passed+1
        mark="PASS"
    else:
        mark="FAIL"
    print(f"{mark}  got={got:9} want={expected:9}  {query}")

print(f"\n{passed}/{len(CASES)} passed")