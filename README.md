# chromadb-llama3-rag
This project implements a local question-answering system using the Llama 3 language model, ChromaDB for storing and retrieving document embeddings, and the Rich library for a user-friendly command-line interface.

## Features

*   **Local Processing:** Runs entirely on your machine, no external API dependencies after initial setup.
*   **Llama 3:** Leverages the power of the Llama 3 language model for question answering.
*   **ChromaDB:** Uses ChromaDB for efficient storage and retrieval of document embeddings.
*   **Rich CLI:** Provides a visually appealing and interactive command-line interface with colored output and panels.
*   **Command History:**  Uses `readline` to provide command history (up/down arrow keys) for a better user experience.
*   **Document Chunking:** Splits large documents into smaller chunks for better retrieval.

## Requirements

*   Python 3.7+
*   CUDA-enabled GPU (recommended)

## Installation

1.  Clone this repository:

    ```bash
    git clone https://github.com/tmeftah/chromadb-llama3-rag
    cd chromadb-llama3-rag
    ```

2.  Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```
    
3.  Download the necessary models:

    The script uses `unsloth/Llama-3.2-3B-Instruct` by default. You may need to accept the license agreement or configure access tokens if you're using models from Hugging Face Hub.

## Usage

1.  Prepare your document.  The default file path to upload is `qa.txt`. Each paragraph should be separated by two newlines (`\n\n`).

2.  Run the script:

    ```bash
    python main.py --qa-file qa.txt
    ```

    Replace `your_script_name.py` with the actual name of your Python script. And `qa.txt` with the path to your document file. You can also customize the model used:

        ```bash
        python main.py --model-name "meta-llama/Llama-3-8B-Instruct"
        ```

3.  Ask your questions!  Type your question and press Enter. Use the up/down arrow keys to access command history. Type `exit` to quit.

## Customization

*   **Models:** You can change the `model_name` and `embed_model_name` parameters to use different models.
*   **Documents:**  Modify the `qa_file` parameter to use a different document file.
*   **ChromaDB Collection:** Change the `collection_name` parameter to use a different ChromaDB collection.
*   **Border Color:** Update the `BORDER_COLOR` constant in the script to change the color of the panels.

## License

