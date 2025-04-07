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


def query_db(question, collection, embed_tokenizer, embed_model, top_k=4):
    """Queries the ChromaDB collection for relevant documents."""
    q_embeddings = get_embeddings([question], embed_tokenizer, embed_model)
    results = collection.query(query_embeddings=q_embeddings.numpy().tolist(), n_results=top_k, include=["documents"])
    return results


def initialize_chroma_db(documents, embed_tokenizer, embed_model, collection_name="document_collection"):
    """Initializes Chroma DB with the given documents."""
    client = chromadb.Client()
    try:
        collection = client.get_collection(name=collection_name)
        console.print(f"[bold green]Collection '{collection_name}' already exists. Using existing collection.[/bold green]")
    except :
        collection = client.create_collection(collection_name)
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
            results = query_db(question, collection, embed_tokenizer, embed_model)
            context = '\n\n'.join(results['documents'][0])
            query_db_time = time.time() - start_time

            query = f"""Your task is to determine the estimated working days required to complete the following customer request based on the provided context details:
                    {context}
                    ---------------
                    Customer Request::
                    {question}
                    ---------------
                    Instructions:
                    Use only the context entries that are directly relevant to the customer's request.
                    If needed combine the working day estimates appropriately for the aspects of the request .
                    Provide the total estimated working days along with a brief justification.
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
        console.print(Panel(Markdown(context), title="[bold cyan]Context[/bold cyan]", border_style=BORDER_COLOR))
        #console.print("-" * 40)

        # Loading information
        timing_info = f"""
  [bold blue]Model Load Time:[/bold blue] {model_load_time:.2f} seconds
  [bold blue]Document Load Time:[/bold blue] {document_load_time:.2f} seconds
  [bold blue]Chroma DB Initialization Time:[/bold blue] {chroma_init_time:.2f} seconds
  [bold blue]Query DB Time:[/bold blue] {query_db_time:.2f} seconds
  [bold blue]Answer Generation Time:[/bold blue] {answer_generation_time:.2f} seconds
        """
        console.print(Panel(timing_info, title="[bold yellow]Processing Times[/bold yellow]", border_style=BORDER_COLOR))
        console.print("-" * 40)



if __name__ == "__main__":
    typer.run(main)
