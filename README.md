# basic_agent

**basic_agent** is an AI-powered agent written in C++ that supports memory, retrieval-augmented generation (RAG), embeddings, and modular toolchains.  
It is designed as a foundation for building intelligent assistants that can reason, recall, and take action through pluggable tools and workflows.

---

## ✨ Features

- **Memory System**
  - Persistent JSON-based long-term memory
  - Episodic recall of past interactions
  - RAG indexing and retrieval

- **Retrieval-Augmented Generation (RAG)**
  - Embedding-based search across documents and code
  - Flexible backends (local vector store, external DBs in the future)

- **Toolchain & Workflows**
  - Modular tools: web search, summarization, command execution
  - Extensible workflows (decision trees/roadmaps for solving tasks)
  - Supports chaining tools together (e.g., Search → Summarize → Store in memory)

- **Embeddings**
  - Local embedding engine for C++ code and text
  - Cosine similarity–based vector retrieval

- **File Handling**
  - Handles text (`.txt`), markdown (`.md`), JSON, and project source code
  - Extensible to support PDFs, EPUBs, and more

- **Configurable Environment**
  - `.env` file support for API keys (OpenAI, Google CSE, etc.)
  - CMake-based build system

---

## 📂 Project Structure

```
basic_agent/
├── CMakeLists.txt         # CMake build configuration
├── include/               # Header files
│   ├── command_processor.h
│   ├── embedding_engine.h
│   ├── memory.h
│   ├── rag.h
│   ├── tools.h
│   └── vector_store.h
├── src/                   # Source code
│   ├── command_processor.cpp
│   ├── embedding_engine.cpp
│   ├── main.cpp
│   ├── memory.cpp
│   ├── rag.cpp
│   ├── tools.cpp
│   └── vector_store.cpp
├── agent_workspace/       # Runtime workspace
│   ├── memory.json        # Persistent agent memory
│   └── rag_index.json     # Vector store index
└── README.md              # Project documentation
```

---

## ⚙️ Installation

### Prerequisites

- C++17 or later
- CMake 3.15+
- A modern compiler (GCC, Clang, MSVC)
- (Optional) API keys for web search or external embeddings, stored in `.env`

### Build Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/basic_agent.git
cd basic_agent

# Create build directory
mkdir build && cd build

# Run CMake
cmake ..

# Build
make -j4
```

---

## 🚀 Usage

```bash
./basic_agent
```

On startup, the agent will:

1. Load persistent memory (`agent_workspace/memory.json`)
2. Initialize RAG vector index (`agent_workspace/rag_index.json`)
3. Be ready for input commands

---

## 🛠️ Tools & Workflows

The agent supports modular tools that can be chained into workflows:

- **Web Search Tool**
  - Queries external search APIs
  - Extracts and summarizes results

- **Summarization Tool**
  - Condenses long text into key insights
  - Stores summaries in memory

- **Command Execution Tool**
  - Runs shell commands safely
  - Returns outputs for further processing

### Example Workflow

```
User: "Find me info on vector databases"
Agent:
  1. Uses WebSearchTool with query
  2. Runs SummarizationTool on results
  3. Stores summary into memory
  4. Returns concise answer
```

---

## 📦 Memory & RAG

- **Memory (`memory.json`)**
  - Stores past conversations and important data
  - Structured for retrieval and context injection

- **RAG Index (`rag_index.json`)**
  - Stores vector embeddings of text/code
  - Enables similarity-based search
  - Powered by **cosine similarity**

---

## 🔧 Configuration

Environment variables are loaded from `.env`:

```
OPENAI_API_KEY=your_api_key_here
GOOGLE_CSE_ID=your_cse_id_here
GOOGLE_API_KEY=your_google_api_key_here
```

---

## 🧩 Roadmap

- [x] Persistent memory
- [x] RAG embeddings with cosine similarity
- [x] Toolchain (web search, summarize, execute)
- [ ] PDF/EPUB ingestion
- [ ] Vector database backend (e.g., SQLite + FAISS, Pinecone)
- [ ] Advanced workflows (decision trees, planning)
- [ ] Multi-agent collaboration

---

## 📝 License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome!  
Please fork the repo and submit a pull request.

