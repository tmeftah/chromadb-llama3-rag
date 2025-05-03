def remove_duplicate_documents(input_filepath):
    try:
        with open(input_filepath, "r", encoding="utf-8") as infile:
            raw_text = infile.read()

        documents = raw_text.split("*********************")
        print(len(documents))
        unique_documents = []
        seen_documents = set()

        for doc_original in documents:
            doc = "\n".join(doc_original.split("\n")[6:])  # Extract document content
            doc = doc.strip()  # Remove leading/trailing whitespace
            if doc and doc not in seen_documents:  # Check for content and uniqueness
                unique_documents.append(doc_original)
            
                seen_documents.add(doc)
        return unique_documents,seen_documents
    except FileNotFoundError:
        print(f"Error: Input file '{input_filepath}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
u,d = remove_duplicate_documents("prod.txt")

print(len(u),len(d))
print(list(u)[0])

with open("unique_prod.txt", "w", encoding="utf-8") as outfile:
            for i, doc in enumerate(u):                
                outfile.write(doc) #added newline for each document
                outfile.write("*********************")
