# Project Overview

The objective of this project is to implement a **Retrieval-Augmented Generation (RAG)** pipeline that enables users to interact with semi-structured data extracted from multiple PDF files. By leveraging advanced NLP tools, including **LangChain** and the **Ollama LLM**, the system extracts, processes, and organizes the content for accurate information retrieval. It can answer specific queries and perform comparisons based on data from PDF documents.

The RAG pipeline combines the strengths of vector similarity search and large language models (LLMs) to deliver precise, context-aware responses to natural language queries.

---

# Approach

The system follows a systematic approach for data ingestion, processing, retrieval, and response generation. The key steps include:

## 1. **Data Ingestion**
   - **Input:** One or more PDF files containing semi-structured data.
   - **Process:**
     - Extract text content from PDFs using `UnstructuredPDFLoader`.
     - Split the extracted text into smaller, logical chunks using `RecursiveCharacterTextSplitter` for granularity.
     - Convert these chunks into **vector embeddings** using the `OllamaEmbeddings` model.
     - Store the embeddings in a local **vector database** using **ChromaDB** for fast and efficient retrieval.

   **Code Implementation:**
   ```python
   loader = UnstructuredPDFLoader("data/sample.pdf")
   documents = loader.load()
   
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
   chunks = text_splitter.split_documents(documents)

   embeddings = OllamaEmbeddings()
   db = Chroma.from_documents(chunks, embeddings)
   ```

## 2. **Query Handling**
   - **Input:** A user query in natural language.
   - **Process:**
     - Convert the user query into **vector embeddings**.
     - Perform a **similarity search** in the vector database to retrieve the most relevant document chunks.
     - Pass the retrieved chunks and the user query to the LLM (`ChatOllama`) to generate a natural language response.

   **Code Implementation:**
   ```python
   retriever = db.as_retriever()
   relevant_docs = retriever.get_relevant_documents("What is the unemployment rate for degree holders?")

   llm = ChatOllama()
   prompt_template = ChatPromptTemplate.from_template("Answer the following: {question}")
   chain = prompt_template | llm

   response = chain.invoke({"question": "What is the unemployment rate for degree holders?"})
   print(response.content)
   ```

## 3. **Comparison Queries**
   - **Input:** User queries that require comparing data across multiple PDFs.
   - **Process:**
     - Identify key terms in the user query that require comparison.
     - Perform similarity searches to retrieve relevant chunks from different PDF files.
     - Aggregate and organize the retrieved data for comparison.
     - Generate structured responses, such as tables or bullet points, to display the comparison.

   **Example Query:**  
   _"Compare unemployment rates for different degrees mentioned in the PDF."_  
   The system retrieves relevant data from pages containing degree-related statistics and outputs a structured comparison.

   **Code Implementation (conceptual):**
   ```python
   comparison_query = "Compare unemployment rates for different degrees"
   retrieved_chunks = retriever.get_relevant_documents(comparison_query)

   # Aggregate and format retrieved information
   formatted_response = format_comparison_results(retrieved_chunks)

   print(formatted_response)
   ```

## 4. **Response Generation**
   - **Input:** User query and relevant document chunks retrieved from the vector database.
   - **Process:**
     - Pass the retrieved content as context to the LLM.
     - Ensure factual and precise answers by integrating retrieved content into the response.
     - For comparison queries, structure the output in a user-friendly format (e.g., tables, lists).

   **Example Output:**

   **Query:** "What is the unemployment rate for individuals with a bachelor's degree?"  
   **Response:**
   ```
   Based on the data from the PDF (Page 2):
   - The unemployment rate for individuals with a Bachelor's degree is 5.6%.
   ```

   **Query:** "Summarize the tabular data from Page 6."  
   **Response:**
   ```
   The table on Page 6 presents the following data:

   | Degree Type           | Unemployment Rate | Median Salary  |
   |-----------------------|------------------:|---------------:|
   | High School Diploma   | 8.1%             | $35,000        |
   | Associate's Degree    | 6.8%             | $42,000        |
   | Bachelor's Degree     | 5.6%             | $60,000        |
   ```

---

# Outcomes

The RAG pipeline delivers the following outcomes:

1. **Accurate Query Responses:**
   - The system extracts precise answers from PDFs by combining vector search with LLM-generated responses.
   - Example: Extract unemployment data for a specific degree type.

2. **Efficient Data Retrieval:**
   - Text is split into optimized chunks and stored as embeddings for fast similarity-based searches.
   - Ensures low latency in retrieving relevant content.

3. **Structured Comparisons:**
   - For comparison queries, the system aggregates and formats data into structured outputs like tables or bullet points.

4. **Interactive User Experience:**
   - Users can "chat" with PDFs by asking questions naturally, similar to interacting with a human expert.
   - Example: "Summarize the table on Page 6" or "What are the key statistics for unemployment rates?"

5. **Scalability:**
   - The system can process multiple PDFs simultaneously and handle queries across large document datasets.

---

#Input and Outputs

**Input 1:**  
_"What is the unemployment rate for individuals with a Bachelor's degree?"_

**Output 1:**  
_Based on the data from the PDF (Page 2), the unemployment rate for individuals with a Bachelor's degree is **5.6%**._

**Input 2:**  
_"Summarize the table on Page 6."_

**Output 2:**  
```
| Degree Type           | Unemployment Rate | Median Salary  |
|-----------------------|------------------:|---------------:|
| High School Diploma   | 8.1%             | $35,000        |
| Associate's Degree    | 6.8%             | $42,000        |
| Bachelor's Degree     | 5.6%             | $60,000        |
```

**Input 3:**  
_"Compare unemployment rates for High School Diploma and Bachelor's Degree holders."_

**Output 3:**  
```
Comparison of Unemployment Rates:
- High School Diploma: 8.1%
- Bachelor's Degree: 5.6%
```
