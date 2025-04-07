import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
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

app = typer.Typer()
console = Console()

BORDER_COLOR = "bright_blue"  # Define a constant for the border color

HISTORY_FILE = os.path.expanduser("~/.qa_history")  # Path to the history file


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


def query_db(question, collection, embed_tokenizer, embed_model, threshold=0.6, top_k=4):
    """Queries the ChromaDB collection for relevant documents."""
    q_embeddings = get_embeddings([question], embed_tokenizer, embed_model)
    results = collection.query(query_embeddings=q_embeddings.numpy().tolist(), n_results=top_k, include=["documents", "distances"])
    documents = results['documents'][0]
    distances = results['distances'][0]

    # Filter based on relevance
    relevant_documents = []
    relevant_distances = []
    for i, doc in enumerate(documents):
      if distances[i] <= threshold:  # Higher similarity score means better relevance
          relevant_documents.append(doc)
          relevant_distances.append(distances[i])

    return relevant_documents, relevant_distances


def initialize_chroma_db(documents, embed_tokenizer, embed_model, collection_name="document_collection"):
    """Initializes Chroma DB with the given documents."""
    client = chromadb.Client()
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
    return raw_text.split("\n\n")


def load_history():
    """Loads command history from the history file."""
    try:
        readline.read_history_file(HISTORY_FILE)
    except FileNotFoundError:
        pass


def save_history():
    """Saves command history to the history file."""
    readline.write_history_file(HISTORY_FILE)


@app.command()
def main(
        model_name: str = "unsloth/Llama-3.2-3B-Instruct",
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        qa_file: str = "qa.txt",
        collection_name: str = "document_collection"
):
    """Main function to run the QA system."""

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

    while True:
        # Beautifully formatted input
        try: # Add try-except block to gracefully handle KeyboardInterrupt
            question = console.input("[bold magenta]> [/bold magenta][bold]What is your question? (Type 'exit' to quit):[/bold] ")
        except KeyboardInterrupt:
            console.print("\n[bold red]Exiting...[/bold red]")  # Print message on Ctrl+C
            break

        if question.lower() == "exit":
            console.print("[bold red]Exiting...[/bold red]")
            break

        with console.status("[bold blue]Processing your request...[/bold blue]") as status:
            # Querying database
            start_time = time.time()
            relevant_documents, relevant_distances = query_db(question, collection, embed_tokenizer, embed_model)
            context = '\n\n'.join(relevant_documents)            
            query_db_time = time.time() - start_time

            if not context.strip():  # Check if the context is empty
                answer = "I'm sorry, I cannot provide an estimate as there is no relevant information available in the provided context."
                answer_generation_time= 0.0
            else:
                query = f"""Your task is to determine the estimated working days required to complete the following customer request based on the provided context details:
                {context}
                ---------------
                Customer Request::
                {question}
                ---------------
                - If the provided context is empty or does not contain directly relevant information for the customer's request, respond with: 
                    "I'm sorry, I cannot provide an estimate as there is no relevant information available in the provided context."
                - If the context contains relevant information, follow the steps below:
                1. Extract only the entries from the context that are directly relevant to the customer's request.
                2. Combine the working day estimates appropriately for the aspects of the request, if necessary.
                3. Format your response like this:
                    **Estimated Total Working Days**: X days  
                    **Justification**: [Provide a brief explanation for how the estimate was calculated based on the relevant entries from the context.]
                    **Context Used**:
                   [List each relevant item from the context along with the respective working days (e.g., "Task A requires 3 working days"). Include only the relevant parts.]
                """
            

                # Generating answer
                start_time = time.time()
                messages = [
                    {"role": "user", "content": query},
                ]
                result = qa_model(messages, max_new_tokens=512, temperature=0.1, top_p=0.95)
                answer = result[0]["generated_text"][1]["content"]
                answer_generation_time = time.time() - start_time


        # Display the results with rich:
        console.print(Panel(question, title="[bold magenta]Question[/bold magenta]", border_style=BORDER_COLOR))
        console.print(Panel(Markdown(answer), title="[bold green]Answer[/bold green]", border_style=BORDER_COLOR))

        # Format the context with distances
        context_with_distances = ""
        for i, doc in enumerate(relevant_documents):
            context_with_distances += (
                f"[bold cyan]Document {i+1}:[/bold cyan] {doc}\n"
                f"[bold yellow]Distance:[/bold yellow] [green]{relevant_distances[i]:.4f}[/green]\n\n"
            )

        console.print(Panel(context_with_distances, title="[bold cyan]Context with Distances[/bold cyan]", border_style=BORDER_COLOR))
        #console.print("-" * 40)

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
