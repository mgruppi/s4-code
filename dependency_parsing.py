import spacy

text = ["Net income was $9.4 million compared to the prior year of $2.7 million.",
        "$9.4 million was the net income, compared to the prior year of $2.7 million."]

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("merge_entities")
nlp.add_pipe("merge_noun_chunks")

for doc in nlp.pipe(text):
    print("---")
    print(doc)
    for token in doc:
        if token.ent_type_ == "MONEY":
            # Attribute and direct object, check for subject
            if token.dep_ in ("attr", "dobj"):
                subj = [w for w in token.head.lefts if w.dep_ == "nsubj"]
                if subj:
                    print(subj[0], "-->", token)
            # We have a prepositional object with a preposition
            elif token.dep_ == "pobj" and token.head.dep_ == "prep":
                print(token.head.head, "-->", token)
