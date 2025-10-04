# Mary AI Agent

Mary AI Agent is a lightweight intelligent product search and conversation system that combines vector retrieval (FAISS + SentenceTransformer) with natural language interaction.  
It can understand user input, perform semantic search, image retrieval, text + image hybrid retrieval, and natural conversations.  
The entire project runs completely offline without any external API.

---

## Project Overview

The goal of Mary AI Agent is to build a lightweight, extensible local AI assistant for:
- Semantic understanding-based product search  
- Image similarity matching  
- Natural language conversation and Q&A  
- Fully offline and local operation  

Users can interact with the AI through the command line or use the Streamlit front-end to experience the chat-based search interface.

---

## Module Description

| Module | Description |
|--------|-------------|
| `product_loader.py` | Loads and reads product information (products.json). |
| `vector_store.py` | Builds the FAISS vector database for text and image feature retrieval. |
| `commerce_agent.py` | Core AI logic: handles text, image, and mixed queries. |
| `front_app.py` | Streamlit front-end interface with chat-style interaction. |
| `api.py` | Modular interface that can be extended into an independent API service. |
| `data/images/` | Stores example product images. |
| `products.json` | Product information database (name, price, image path, description, etc.). |

---

## Example Interactions

### Text Search  
**Input:**  
`Enter query: running shoes`  

**Output:**  
Returns products related to running shoes and their prices and ratings.

---

### Image Search  
**Input:**  
`Enter query: backpack.jpg`  

**Output:**  
Returns products most similar to the uploaded image.

---

### Mixed Search  
**Input:**  
`Enter query: black jacket | leather_jacket.jpg`  

**Output:**  
Combines text and image for hybrid similarity search.

---

### Conversation Example  
**Input:**  
`Enter query: what's your name`  

**Output:**  
Mary: I’m Mary, your AI Agent here to help you with products.

---

## How to Run

### Method 1: Command Line
```bash
python -m agent.commerce_agent
```

### Method 2: Streamlit Front End
```bash
streamlit run front_app.py
``` 
Then open the local page (default: http://localhost:8501)  
to start chatting with Mary in your browser.  
You can type text, provide an image filename (e.g. backpack.jpg),  
or use a combined query like black jacket | leather_jacket.jpg.

---

### Environment Setup
Project dependencies are recorded in `requirements.txt`.  
Install them with:

```bash
pip install -r requirements.txt
``` 

If Streamlit or FAISS cannot be found, make sure you’re using Python 3.9 or above  
and have the required libraries installed.

---

### Summary
Mary AI Agent is a fully offline, lightweight conversational product search assistant. It integrates semantic embeddings, FAISS indexing, and a simple Streamlit UI  
to enable both intelligent search and natural conversation -- all running locally, without any cloud APIs or external dependencies.  

This project shows how a small AI system can combine retrieval and conversation capabilities in one package, with clear structure and fully reproducible local execution.
