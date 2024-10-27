# Self-Corrective Retrieval-Augmented Generation (SCRAG)

This repository presents a Self-Corrective Retrieval-Augmented Generation (SCRAG) framework designed for accurate and context-aware question answering. Using reranking and judgmental filtering, SCRAG refines retrieval results to deliver precise answers from a knowledge base. The repository includes two variants:

1. **SCRAG without Memory**: A straightforward SCRAG implementation focused on retrieving and ranking relevant information.
2. **SCRAG with Memory**: An enhanced version that retains a conversational history, making it adaptable to follow-up questions.

## Key Features

- **Self-Corrective Pipeline**: SCRAG iterates over retrieved results, using reranking and judging to improve answer relevance.
- **Memory Integration**: The memory-enabled version stores conversation history for contextual, follow-up responses.
- **Local Deployment with Ollama**: Using Ollama’s API, SCRAG can run fully locally, ensuring data privacy and fast performance.

## Repository Contents

- `SCRAG_no_memory.ipynb`: Implementation of SCRAG without memory, ideal for standalone queries.
- `SCRAG_with_memory.ipynb`: Memory-enabled SCRAG for context-aware Q&A.
- `setup_ollama.sh`: Shell script to set up Ollama on your local environment or Kaggle.
- **Kaggle Notebook**: Includes both memory and non-memory implementations for easy comparison and experimentation.

## Getting Started

### Prerequisites

- **Ollama**: Ensure Ollama is installed to run the local language model (`llama3`). 
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```

- **Dependencies**: Install required Python libraries in your environment.
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. **Vector Database Creation**:
    Use `create_vector_database()` to preprocess documents, chunk and embed text, and store embeddings in a Chroma database for efficient retrieval.

2. **Running SCRAG**:
   - **Without Memory**: Use `CR(query)` for standard SCRAG operations.
   - **With Memory**: Use `CR_mem(query, memory_buffer)` to retain and build context from past queries.

3. **Deploy Locally**: To fully run SCRAG locally, execute with Ollama, ensuring private, offline Q&A processing.

### Sample Commands

1. Clone this repository.
2. Run `SCRAG_no_memory.ipynb` or `SCRAG_with_memory.ipynb` notebooks on Kaggle or your local environment.

## Contributions

Feel free to open issues or submit pull requests to contribute to this project. Suggestions for improvements or additional features are welcome!

