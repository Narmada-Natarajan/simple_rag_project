# 🤖 Local RAG Employee Handbook Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot that answers questions about an employee handbook PDF using local AI models.

## 🛠️ Tech Stack
- Python
- FAISS – Vector similarity search
- Sentence Transformers – Text embeddings
- Ollama (Llama 3.2) – Answer generation
- PyPDF – PDF text extraction
- Pickle – Store document chunks

## ✨ Features
- 📄 Extracts text from PDF documents.
- Splits text into smaller chunks.
- Generates embeddings locally.
- Retrieves relevant information using FAISS.
- Generates answers using Ollama.
- Displays source page references.
- Works without OpenAI API credits.

## ⚙️ Installation

1. Clone the repository.
2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate it on Windows:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Install [Ollama](https://ollama.com/download/windows) and download the model:

   ```bash
   ollama pull llama3.2
   ```

6. Place your PDF in the project folder as `dummy_employee_handbook_rag.pdf`.

## 🚀 Run the Project

```bash
python main.py
```

Ask questions about the handbook in the terminal. Type `exit` to quit.

## 🔄 How It Works

PDF → Text Chunking → Embeddings → FAISS Retrieval → Ollama → Answer

## 📝 Note

This project is built for learning and demonstration. The employee handbook is a fictional sample, not a real company policy document.
