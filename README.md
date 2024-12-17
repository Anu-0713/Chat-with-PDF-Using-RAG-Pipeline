# Ingesting PDF

## Description

This notebook demonstrates how to process and analyze PDF documents using **vector embeddings** and LLMs (Large Language Models) with `LangChain`, `Ollama`, and other supporting libraries. The workflow includes:

- Loading PDF documents.
- Splitting text into manageable chunks.
- Generating vector embeddings for semantic understanding.
- Storing embeddings in a local vector database.
- Querying and retrieving relevant document segments.
- Leveraging LLMs to interact with your data.

## Features

- **PDF Loading:** Load and parse PDF content seamlessly.
- **Text Chunking:** Split extracted text into optimized chunks for embeddings.
- **Vector Embeddings:** Use `OllamaEmbeddings` to encode document chunks for semantic search.
- **Vector Database:** Store embeddings locally using `ChromaDB`.
- **Querying Documents:** Perform multi-query retrieval to fetch relevant data.
- **LLM Integration:** Use `ChatOllama` to interact with retrieved document sections.

## Requirements

Before running the notebook, ensure you have the following prerequisites:

### Dependencies

Install the required libraries using `pip`:

```bash
pip install langchain langchain_community langchain_ollama langchain_text_splitters chromadb unstructured tqdm ipython
```

### Other Requirements

- **Ollama:** Ensure you have the `ollama` model installed locally.
  - Installation instructions: [Ollama Official Website](https://ollama.ai)
  - Download models like `llama2` or any compatible LLM supported by Ollama.

- **Python 3.8+** is recommended.

## Step-by-Step Workflow

### 1. Load PDF Document

The notebook starts by loading a PDF using `UnstructuredPDFLoader` from `LangChain Community`.

```python
from langchain_community.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader("your_document.pdf")
documents = loader.load()
```

### 2. Split Text into Chunks

To prepare the PDF content for embedding, the text is split into smaller chunks.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
```

### 3. Generate Vector Embeddings

Use `OllamaEmbeddings` to generate embeddings for each text chunk.

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings()
```

### 4. Store Embeddings in a Vector Database

Store the embeddings locally using `ChromaDB`.

```python
from langchain_community.vectorstores import Chroma

db = Chroma.from_documents(chunks, embeddings)
```

### 5. Query the Document

Perform a multi-query search to retrieve relevant chunks based on user input.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever(vectorstore=db)
retrieved_docs = retriever.get_relevant_documents("Your query here")
```

### 6. Chat with the Content Using LLMs

Use `ChatOllama` to generate responses based on the retrieved document segments.

```python
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate

llm = ChatOllama()
prompt = ChatPromptTemplate.from_template("Answer the following based on the document: {question}")
chain = prompt | llm

response = chain.invoke({"question": "Your question here"})
print(response.content)
```

## Example Use Case

Imagine you have a PDF document containing research papers, reports, or manuals. You can:

1. **Load** the document.
2. **Query** the content for specific information such as:
   - "Summarize the document in 3 sentences."
   - "What are the key findings related to X?"
   - "Provide a step-by-step explanation of process Y."

3. **Chat** interactively with the content as if you were asking an expert.

## Running the Notebook Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required dependencies:

   ```bash
   pip install langchain langchain_community langchain_ollama langchain_text_splitters chromadb unstructured tqdm ipython
   ```

3. Ensure you have **Ollama** installed and running:

   ```bash
   ollama run llama2
   ```

4. Launch the Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open the `local_ollama_rag.ipynb` notebook and follow the steps.

## Folder Structure

```plaintext
your-repo-name/
│
├── local_ollama_rag.ipynb   # The main notebook
├── README.md                # Project documentation
└── data/                    # Add your PDF files here
```

## Future Improvements

- Add support for other document formats (e.g., DOCX, TXT).
- Integrate cloud-based vector databases.
- Expand to use other LLM providers.

---

Let me know if you’d like more tweaks or additions! 🚀
