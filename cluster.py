import os
import time
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich import print as rprint

import typer
from typing_extensions import Annotated

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import re  # Import the regular expression module


console = Console()


def load_data(file_path: Path) -> str:
    """Loads the data from the provided file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
        return data
    except FileNotFoundError:
        console.print_exception()
        raise typer.Exit(code=1)


def embed_data(data: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Embeds the data using the specified sentence transformer model."""
    console.log(f"Embedding data with model: {model_name}")
    model = SentenceTransformer(model_name)
    # Split the data into sentences.  A very simple approach.
    sentences = data.split("\n\n")  # This can be improved.
    sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty strings
    embeddings = model.encode(sentences)
    console.log(f"Data embedded into {embeddings.shape}")
    return embeddings, sentences


def reduce_dimensionality(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduces the dimensionality of the embeddings using t-SNE."""
    console.log(f"Reducing dimensionality to {n_components} components using t-SNE")
    tsne = TSNE(n_components=n_components, perplexity=15, max_iter=300, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    console.log(f"Dimensionality reduced to {reduced_embeddings.shape}")
    return reduced_embeddings


def cluster_data(reduced_embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """Clusters the reduced embeddings using KMeans."""
    console.log(f"Clustering data into {n_clusters} clusters using KMeans")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # Explicitly set n_init
    clusters = kmeans.fit_predict(reduced_embeddings)
    console.log(f"Data clustered.")
    return clusters


def plot_clusters(reduced_embeddings: np.ndarray, clusters: np.ndarray, sentences: list, cluster_topics: dict, output_path: Path = Path("cluster_plot.png")):
    """Plots the clusters using matplotlib."""
    console.log(f"Plotting clusters and saving to {output_path}")
    df = pd.DataFrame(
        dict(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            label=clusters,
            text=sentences,
        )
    )
    groups = df.groupby("label")

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.margins(0.05)

    for name, group in groups:
        # Use cluster topic as legend label if available, otherwise cluster number
        topic = cluster_topics.get(name, f"Cluster {name}")
        ax.plot(
            group.x,
            group.y,
            marker="o",
            linestyle="",
            label=topic,
            ms=12,
        )
        # Optionally add annotations (first 5 points of each group)
        for i in range(min(5, len(group))):
            ax.text(
                group.iloc[i].x,
                group.iloc[i].y,
                group.iloc[i].text[:50] + "...",  # Show first 50 characters
                fontsize=8,
                alpha=0.7,
            )

    ax.legend()
    ax.set_title("t-SNE Visualization of Sentence Embeddings")
    plt.savefig(output_path)
    console.log(f"Plot saved to {output_path}")


def export_csv(reduced_embeddings: np.ndarray, clusters: np.ndarray, sentences: list, cluster_topics: dict, csv_path: Path = Path("cluster_data.csv")):
    """Exports the cluster data to a CSV file."""
    console.log(f"Exporting cluster data to {csv_path}")

    # Replace newlines in sentences with spaces
    cleaned_sentences = [s.replace('\n', ' ') for s in sentences]

    df = pd.DataFrame(
        {
            "x": reduced_embeddings[:, 0],
            "y": reduced_embeddings[:, 1],
            "cluster": clusters,
            "topic": [cluster_topics.get(cluster, "Unknown Topic") for cluster in clusters], # Added topic column
            "text": cleaned_sentences,
        }
    )
    df.to_csv(csv_path, index=False, encoding="utf-8", sep=';', lineterminator='\n')
    console.log(f"Cluster data exported to {csv_path}")


def get_cluster_topics(sentences: list, clusters: np.ndarray, model_name: str = "unsloth/Llama-3.2-3B-Instruct"):
    """
    Identifies the topic of each cluster by sending the cluster's documents to an LLM.
    """
    console.log(f"Getting cluster topics using LLM: {model_name}")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        

    except Exception as e:
        console.print_exception()
        console.print(f"[red]Error loading LLM model: {e}.  Make sure you have enough resources (RAM/GPU) and the correct model name.[/]")
        return {}

    cluster_topics = {}
    unique_clusters = np.unique(clusters)


    for cluster_id in unique_clusters:
       
        console.log(f"Analyzing cluster: {cluster_id}")
        cluster_sentences = [sentences[i] for i, cluster in enumerate(clusters) if cluster == cluster_id]
        cluster_text = "\n".join(cluster_sentences)  # Combine sentences for the prompt

        # Construct the prompt.  Adjust the prompt for better results.
        prompt = f"{cluster_text}\n\n Please give us only one text name. Follow this format: **Text header** and ignore  working days "

        try:
            result = pipe(prompt,max_new_tokens=200, temperature=0.3, top_p=0.98) # Adjust max_length as needed

            if result:
                generated_text = result[0]['generated_text'].replace(prompt, "").strip()

                 # Extract document name using regex
                match = re.search(r"\*\*(.*?)\*\*", generated_text)  # Modified regex
                if match:
                    topic = match.group(1).strip()
                else:
                    topic = "No text header found."

                cluster_topics[cluster_id] = topic
                console.log(f"Cluster {cluster_id} topic: {topic}")
                print(20*"*")
            else:
                cluster_topics[cluster_id] = "No topic identified."
                console.log(f"Cluster {cluster_id}: No topic identified.")

        except Exception as e:
            console.print_exception()
            console.log(f"[red]Error generating topic for cluster {cluster_id}: {e}[/]")
            cluster_topics[cluster_id] = "Error generating topic."

    return cluster_topics


def main(
    file_path: Annotated[Path, typer.Option(help="Path to the input text file.")] = Path("qa.txt"),
    model_name: Annotated[str, typer.Option(help="Name of the sentence transformer model.")] = "sentence-transformers/all-MiniLM-L6-v2",
    n_clusters: Annotated[int, typer.Option(help="Number of clusters for KMeans.")] = 5,
    output_path: Annotated[Path, typer.Option(help="Path to save the cluster plot.")] = Path("cluster_plot.png"),
    csv_path: Annotated[Path, typer.Option(help="Path to save the cluster data as CSV.")] = Path("cluster_data.csv"),
    llm_model_name: Annotated[str, typer.Option(help="Name of the LLM model for topic extraction.")] = "unsloth/Llama-3.2-3B-Instruct",
):
    """
    Embeds text data from a file using a sentence transformer model, reduces
    dimensionality with t-SNE, clusters the data with KMeans, plots the clusters,
    exports the data to a CSV file, and identifies the topic of each cluster using an LLM.
    """
    console.rule("[bold blue]Text Embedding and Clustering with Topic Extraction[/]")

    try:
        data = load_data(file_path)
        embeddings, sentences = embed_data(data, model_name)
        reduced_embeddings = reduce_dimensionality(embeddings)
        clusters = cluster_data(reduced_embeddings, n_clusters)
        cluster_topics = get_cluster_topics(sentences, clusters, llm_model_name)  # Get cluster topics BEFORE plotting and exporting
        plot_clusters(reduced_embeddings, clusters, sentences, cluster_topics, output_path)  # Pass cluster topics to plotting function
        export_csv(reduced_embeddings, clusters, sentences, cluster_topics, csv_path)  # Pass cluster topics to export function

        console.print(Panel(f"[green]Successfully processed data, created plot: {output_path} and CSV: {csv_path}[/]"))
        console.print("[bold green]Cluster Topics:[/]")
        for cluster_id, topic in cluster_topics.items():
            console.print(f"[bold]Cluster {cluster_id}:[/] {topic}")

    except Exception as e:
        console.print_exception()
        console.print(f"[red]An error occurred: {e}[/]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
