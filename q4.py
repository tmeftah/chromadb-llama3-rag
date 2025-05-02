import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import CrossEncoder
from transformers import pipeline
import chromadb
import typer
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
import time
import os
import atexit
import readline
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
app = typer.Typer()
console = Console()

BORDER_COLOR = "bright_blue"  # Define a constant for the border color

HISTORY_FILE = os.path.expanduser("~/.qa_history")  # Path to the history file


# ChromaDB settings
settings=Settings(is_persistent=True)
client = chromadb.Client(settings=settings,tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,)

# Load the model, here we use our base sized model
reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1")

def load_models(model_name="unsloth/Llama-3.2-3B-Instruct", embed_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Loads the tokenizer, embedding model, and QA model."""
    embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
    embed_model = AutoModel.from_pretrained(embed_model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_model = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return embed_tokenizer, embed_model, qa_model, tokenizer, model


def get_embeddings(texts, embed_tokenizer, embed_model):
    """Generates embeddings for a list of texts."""
    inputs = embed_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = embed_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings


def query_db(question, collection, embed_tokenizer, embed_model, threshold=0.6, top_k=10):
    """Queries the ChromaDB collection for relevant documents."""
    q_embeddings = get_embeddings([question], embed_tokenizer, embed_model)
    results = collection.query(query_embeddings=q_embeddings.numpy().tolist(), n_results=top_k, include=["documents", "distances"])
    documents = results['documents'][0]
    distances = results['distances'][0]
    #print(distances)

    # Filter based on relevance
    relevant_documents = []
    relevant_distances = []
    for i, doc in enumerate(documents):
      if abs(distances[i]) <= threshold:  # Higher similarity score means better relevance
          relevant_documents.append(doc)
          relevant_distances.append(abs(distances[i]))

    return relevant_documents, relevant_distances


def initialize_chroma_db(documents, embed_tokenizer, embed_model, collection_name="document_collection"):
    """Initializes Chroma DB with the given documents."""
    
    try:
        collection = client.get_collection(name=collection_name)
        console.print(f"[bold green]Collection '{collection_name}' already exists. Using existing collection.[/bold green]")
    except :
        collection = client.create_collection(collection_name, metadata={
        "hnsw:space": "cosine",
        "hnsw:search_ef": 100
    })
        console.print(f"[bold green]Creating new collection: '{collection_name}'[/bold green]")

        # Process each chunk: embed it and add it to the ChromaDB collection.
        for idx, chunk in enumerate(documents):
            embedding = get_embeddings(chunk, embed_tokenizer, embed_model)
            collection.add(
                ids=[f"chunk_{idx}"],
                documents=[chunk],
                embeddings=embedding.tolist()
            )

    return collection


def load_documents(file_path="qa.txt"):
    """Loads documents from a text file, splitting them into chunks."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return raw_text.split("*********************")


def load_history():
    """Loads command history from the history file."""
    try:
        readline.read_history_file(HISTORY_FILE)
    except FileNotFoundError:
        pass


def save_history():
    """Saves command history to the history file."""
    readline.write_history_file(HISTORY_FILE)

def add_document_to_db_if_not_exists(document, collection, embed_tokenizer, embed_model, distance_threshold=0.05):
    """Adds a document to the ChromaDB collection only if it doesn't already exist (based on distance)."""

    q_embeddings = get_embeddings([document], embed_tokenizer, embed_model)
    results = collection.query(
        query_embeddings=q_embeddings.numpy().tolist(),
        n_results=1,  # Check only the most similar document
        include=["documents", "distances"]
    )

    if results['documents'] and results['documents'][0] and results['distances'] and results['distances'][0]:
        existing_document = results['documents'][0][0]
        distance = results['distances'][0][0]
        distance = round(distance, 2)  # Round the distance to 2 decimal places

        if distance <= distance_threshold:
            console.print(f"[bold yellow]Document already exists in the database (distance: {distance:.2f}). Skipping addition.[/bold yellow]")
            return False  # Document already exists

    # If the document doesn't exist or isn't similar enough, add it.
    embedding = get_embeddings([document], embed_tokenizer, embed_model)
    new_id = f"added_doc_{int(time.time())}"  # Create a unique ID.
    collection.add(
        ids=[new_id],
        documents=[document],
        embeddings=embedding.tolist()
    )
    console.print(f"[bold green]Document added to collection with ID: {new_id}[/bold green]")
    return True # Document added

@app.command()
def main(
        model_name: str = "unsloth/Llama-3.2-3B-Instruct",
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        qa_file: str = "qa.txt",
        collection_name: str = "document_collection",
        distance_threshold: float = 0.05 , # new param
        show_context: bool=False,
):
    """Main function to run the QA system within the context of a rich panel."""

    # Enhanced header panel
    header_panel = Panel(
        "[bold yellow]Welcome to Aufwandrechner system![/bold yellow]\n\n"
        "[bold green]How it works:[/bold green]\n"
        "- Ask your question about working day estimates for specific tasks or objectives.\n"
        "- The system searches through the context and provides estimates based on relevant information.\n"
        "- The system will now prevent duplicated documents from being added to the knowledge base.\n\n"
        "[bold cyan]Examples of questions you can ask:[/bold cyan]\n"
        "- \"How many working days are required for task X?\"\n"
        "- \"What is the estimated time for project Y?\"\n\n"
        "[bold red]Other Commands:[/bold red]\n"
        "- 'add_doc' - Add a new document to the knowledge base (only if it doesn't already exist).\n"  # Updated doc string
        "- If you need to exit, type [bold]'exit'[/bold] or press [bold]CTRL+C[/bold].\n\n"
        "[bold blue]Have a productive session! �[/bold blue]",
        border_style=BORDER_COLOR,
        title="[bold cyan]Aufwandrechner[/bold cyan]"

    )
    console.print(header_panel)

    # Load history at startup
    load_history()
    atexit.register(save_history)  # Save history on exit

    # Loading models
    with console.status("[bold blue]Loading models...[/bold blue]") as status:
        start_time = time.time()
        embed_tokenizer, embed_model, qa_model, tokenizer, model = load_models(model_name, embed_model_name)
        model_load_time = time.time() - start_time
        status.update("[bold green]Models loaded.[/bold green]")
        time.sleep(0.3)

    # Loading documents
    with console.status("[bold blue]Loading documents...[/bold blue]") as status:
        start_time = time.time()
        documents = load_documents(qa_file)
        document_load_time = time.time() - start_time
        status.update("[bold green]Documents loaded.[/bold green]")
        time.sleep(0.3)

    # Initializing Chroma DB
    with console.status("[bold blue]Initializing Chroma DB...[/bold blue]") as status:
        start_time = time.time()
        collection = initialize_chroma_db(documents, embed_tokenizer, embed_model, collection_name)
        chroma_init_time = time.time() - start_time
        status.update("[bold green]Chroma DB initialized.[/bold green]")
        time.sleep(0.3)

    # App loop for handling questions
    while True:
        try:
            question = console.input("[bold magenta]> [/bold magenta][bold]What is your question? (Type 'exit' to quit, 'add_doc' ):[/bold] ")
        except KeyboardInterrupt:
            console.print("\n[bold red]Exiting...[/bold red]")  # Gracefully handle Ctrl+C
            break

        if question.lower() == "exit":
            console.print("[bold red]Exiting...[/bold red]")
            break
        

        elif question.lower() == "add_doc":
            # Handle adding a new document
            new_document = console.input("[bold green]Enter the document to add:[/bold green] ")
            with console.status("[bold blue]Adding document to DB (checking for duplicates)...[/bold blue]") as status:
                added = add_document_to_db_if_not_exists(new_document, collection, embed_tokenizer, embed_model, distance_threshold) # passing threshold
                if added:
                    status.update("[bold green]Document added (if it was new).[/bold green]")
                else:
                    status.update("[bold yellow]Document was not added (duplicate).[/bold yellow]")
            time.sleep(0.3)
            continue  # Back to main loop
        else:

            with console.status("[bold blue]Processing your request...[/bold blue]") as status:
                # Querying database
                start_time = time.time()
                relevant_documents, relevant_distances = query_db(question, collection, embed_tokenizer, embed_model,threshold=0.8)
                context = '\n\n'.join(relevant_documents)
                query_db_time = time.time() - start_time

                if not context.strip():  # Check if the context is empty
                    answer = "I'm sorry, I cannot provide an estimate as there is no relevant information available in the provided context."
                    answer_generation_time= 0.0
                else:

                    reranked_results = reranker.rank(question, relevant_documents, return_documents=True, top_k=4)
                    #print(reranked_results)
                    context = '\n\n'.join([doc["text"] for doc in reranked_results])
                    query = f"""Your task is to  estimate working days required to complete the following customer request based on the provided context details:                 
Customer Request:
{question}
------------------------------
{context}
------------------------------ 

- If the provided context is empty or does not contain directly relevant information for the customer's request, respond with:
    "I'm sorry, I cannot provide an estimate as there is no relevant information available in the provided context."
- If the context contains relevant information, follow the steps below:
    0. do not show us the context.                   
    1. if necessary combine the working days estimates appropriately for the aspects of the request.
    2. If context contains round about the same information, do not combine the working days.
    3. response only like this:
Total Working Days : X days
"""


                    # Generating answer
                    start_time = time.time()
                    messages = [                        
                        {"role": "user", "content": query},
                    ]
                    result = qa_model(messages, max_new_tokens=512, temperature=0.6, top_p=0.96)
                    answer = result[0]["generated_text"][1]["content"]
                    answer_generation_time = time.time() - start_time


            # Display the results with rich:
            console.print(Panel(question, title="[bold magenta]Question[/bold magenta]", border_style=BORDER_COLOR))
            console.print(Panel(Markdown(answer), title="[bold green]Answer[/bold green]", border_style=BORDER_COLOR))

            # Create a panel for the reranked results
            reranked_results_text = ""
            for i, doc in enumerate(reranked_results):
                reranked_results_text += f"[bold cyan]Rank {i+1}:[/bold cyan] {doc['text']}\n[bold yellow]Score:[/bold yellow] [green]{doc['score']:.4f}[/green]\n\n"  # added score display
            console.print(Panel(reranked_results_text, title="[bold purple]Reranked Results[/bold purple]", border_style=BORDER_COLOR))

            # Format the context with distances
            if show_context:
                context_with_distances = ""
                for i, doc in enumerate(relevant_documents):
                    context_with_distances += (
                        f"[bold cyan]Document {i+1}:[/bold cyan] {doc}\n"
                        f"[bold yellow]Distance:[/bold yellow] [green]{relevant_distances[i]:.2f}[/green]\n\n"
                    )

                console.print(Panel(context_with_distances, title="[bold cyan]Context with Distances[/bold cyan]", border_style=BORDER_COLOR))

            


            # Loading information
            timing_info = Panel(
                f"[bold blue]Model Load Time:[/bold blue] [green]{model_load_time:.2f} seconds[/green]\n"
                f"[bold blue]Document Load Time:[/bold blue] [green]{document_load_time:.2f} seconds[/green]\n"
                f"[bold blue]Chroma DB Initialization Time:[/bold blue] [green]{chroma_init_time:.2f} seconds[/green]\n"
                f"[bold blue]Query DB Time:[/bold blue] [green]{query_db_time:.2f} seconds[/green]\n"
                f"[bold blue]Answer Generation Time:[/bold blue] [green]{answer_generation_time:.2f} seconds[/green]",
                title="[bold yellow]Processing Times[/bold yellow]",
                border_style=BORDER_COLOR,
            )
            console.print(timing_info)
            console.print("-" * 40)


if __name__ == "__main__":
    typer.run(main)
