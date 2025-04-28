
from bertopic import BERTopic
from pathlib import Path

def load_documents(file_path="qa.txt"):
    """Loads documents from a text file, splitting them into chunks."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return raw_text.split("\n\n")



docs = load_documents()
embedding_model= "sentence-transformers/all-MiniLM-L6-v2"
topic_model = BERTopic(embedding_model=embedding_model, verbose=True,min_topic_size=3)
topics, probs = topic_model.fit_transform(docs)





# 6. Analyze the results
print("Number of topics:", len(topic_model.get_topics()))  # Report how many topics were created

# Print the top 10 most frequent topics
print("\nTop concepts per topic:")
for topic in topic_model.get_topic_info().head(10)["Topic"]:
    print(f"Topic {topic}: {topic_model.get_topic(topic)}")

# Access individual topics and their properties.  For example, the most frequent topic:
most_frequent_topic = topic_model.get_topic_info().loc[0, "Topic"] #Always topic -1, and topic index order per count
print(f"\nMost frequent topic (Topic {most_frequent_topic}): {topic_model.get_topic(most_frequent_topic)}")


topic_model.visualize_topics()



topic_model.visualize_hierarchy()


topic_model.visualize_barchart()


