# basic_agent

**basic_agent** is an AI-powered agent written in **C++20** that supports memory, retrieval-augmented generation (RAG), embeddings, and modular toolchains.  
It is designed as a foundation for building intelligent assistants that can reason, recall, and take action through pluggable tools and workflows.  
⚠️ **Note:** Some features are experimental and may not work yet — for example, the web scraping tool can cause faults in certain environments.

---

## ✨ Features

- **Memory System**
  - Persistent SQLite-based episodic memory
  - Transactional and incremental recall of past interactions
  - RAG indexing and retrieval

- **Retrieval-Augmented Generation (RAG)**
  - Embedding-based search across documents and code
  - Pluggable similarity backends (cosine, dot-product, etc.)
  - Flexible design for swapping in external vector DBs in the future

- **LLM Integration**
  - `LLMInterface` abstraction layer
  - Configurable backend (default: Ollama; supports OpenAI)
  - Pulls API keys from `.env` or system environment
  - Uses libcurl for HTTP-based backends

- **Toolchain & Tools**
  - Modular tools: web search, summarization, command execution
  - Tools can be chained (e.g., Search → Summarize → Store in memory)
  - Web scraping helper (experimental — see Known Issues)

- **Embeddings**
  - Local embedding engine (`TfIdf`, `WordHash`, `Simple`, `External`)
  - Vector store with pluggable similarity metrics
  - Configurable thresholds and limits

- **File Handling**
  - Handles text (`.txt`), markdown (`.md`), JSON, and project source code
  - Extensible to support PDFs, EPUBs, and other formats

- **Config & Environment**
  - `.env` loader for API keys and secrets (EnvLoader)
  - `Config` object for runtime parameters: temperature, top_p, max_tokens, similarity threshold, verbosity
  - CMake-based build system (C++20)

---

## 📂 Project Structure

~~~text
basic_agent/
├── CMakeLists.txt         # CMake build configuration
├── include/               # Header files
│   ├── command_processor.h
│   ├── config.h
│   ├── embedding_engine.h
│   ├── env_loader.h
│   ├── llm_interface.h
│   ├── memory.h
│   ├── prompt_factory.h
│   ├── rag.h
│   ├── index_manager.h
│   ├── webscraperTools.h
│   └── vector_store.h
├── src/                   # Source code
│   ├── command_processor.cpp
│   ├── config.cpp
│   ├── embedding_engine.cpp
│   ├── env_loader.cpp
│   ├── llm_interface.cpp
│   ├── main.cpp
│   ├── memory.cpp
│   ├── prompt_factory.cpp
│   ├── rag.cpp
│   ├── index_manager.cpp
│   ├── webscraperTools.cpp
│   └── vector_store.cpp
├── agent_workspace/       # Runtime workspace
│   ├── memory.db        # Persistent agent memory
│   └── rag_index.json     # Vector store index
├── config.json            # Runtime configuration
├── .env.example           # Example env (API keys - DO NOT COMMIT real keys)
└── README.md              # Project documentation
~~~

---

## ⚙️ Installation

### Prerequisites

- **C++20** (required)
- CMake 3.15+ (3.16+ recommended)
- A modern compiler (GCC, Clang, MSVC)
- `nlohmann/json` (header-only)
- libcurl (for HTTP/LLM backends)
- (Optional) API keys for OpenAI / Google CSE stored in `.env`

### Build Steps

~~~bash
# Clone the repository
git clone https://github.com/yourusername/basic_agent.git
cd basic_agent

# Create build directory
mkdir build && cd build

# Configure (explicitly request C++20 if needed)
cmake -DCMAKE_CXX_STANDARD=20 ..

# Build
make -j4
~~~

> Alternatively set `set(CMAKE_CXX_STANDARD 20)` in the top-level `CMakeLists.txt`.

---

## 🚀 Usage

~~~bash
# Run the built agent
./basic_agent
~~~

On startup the agent will:

1. Load runtime configuration (`config.json`)
2. Load `.env` for API keys (if present)
3. Initialize persistent memory (`agent_workspace/memory.db`)
4. Initialize RAG index (`agent_workspace/rag_index.json`)
5. Launch an interactive command loop (see Example Commands below)

---

## 🛠️ Tools & Workflows

The agent is designed around a small set of modular tools which can be combined:

- **Web Search Tool (experimental)**
  - Uses `WebScraperTools` to query and fetch web content
  - Extracts and summarizes search results
  - ⚠️ Experimental: web scraping has caused faults in some runs; see Known Issues

- **Summarization Tool**
  - Condenses long web pages or documents into short summaries
  - Summaries can be stored into memory for later recall

- **Command Execution Tool**
  - Runs shell commands in a controlled way and returns outputs for processing

### Example Workflow

~~~text
User: "Find me info on vector databases"
Agent:
  1. Uses WebSearchTool with query
  2. Summarizes top results with SummarizationTool
  3. Stores condensed summary into Memory
  4. Returns short, actionable answer to the user
~~~

---

## 📦 Memory & RAG

- **Memory (`memory.db`)**
  - Stores past conversations, summaries, and key facts
  - Used by PromptFactory to build context-aware prompts

- **RAG Index (`rag_index.json`)**
  - Stores embeddings of text and code
  - VectorStore and IndexManager provide chunking, saving, and retrieval
  - Retrieval returns top-k relevant `CodeChunk`s for composing prompts

---

## 🔧 Configuration

### `config.json` (example keys)

~~~json
{
  "temperature": 0.7,
  "top_p": 1.0,
  "max_tokens": 512,
  "max_results": 5,
  "similarity_threshold": 0.7,
  "verbosity": 1,
  "allow_web": true,
  "memory_limit_mb": 256,
  "disk_quota_mb": 512
}
~~~

### `.env` (example)

~~~env
OPENAI_API_KEY=sk-xxxx...
GOOGLE_CSE_ID=your_cse_id
GOOGLE_API_KEY=your_google_api_key
~~~

- Place a `.env` in the repo root (or parent) before running to populate environment variables via `EnvLoader`.
- If `.env` is missing, the process will fall back to system environment variables.

---

## ⚠️ Known Issues & Experimental Parts

> Please treat these as **work-in-progress**. If you rely on the agent for important tasks, test carefully.

- **Web scraping/search (WebScraperTools)**  
  - Experimental and may crash or throw parsing errors in certain network/content edge cases.  
  - If scraping causes faults, disable web access by setting `"allow_web": false` in `config.json` or avoid the scrape commands.
  - Debugging tips: enable higher verbosity (`"verbosity": 2`) and inspect `src/webscraperTools.cpp` for parsing error handling.

- **External API integrations (YouTube, Reddit, News)**  
  - Starter implementations exist but require API keys and more robust error handling.

- **Embedding backends**  
  - TF-IDF is the default local implementation — external/backed models are planned but may require additional configuration.

- **Platform quirks**  
  - `.env` loading uses `setenv` on POSIX and `_putenv_s` on Windows. Behavior may vary with shells/CI.

If you encounter a bug, please open an issue with:
- Steps to reproduce
- `config.json` (redact secrets)
- Any stack trace or logs (increase verbosity if needed)

---

## 🧩 Roadmap

- [x] Persistent memory
- [x] RAG embeddings with pluggable similarity
- [x] Toolchain: web search, summarization, command execution (web scraping experimental)
- [x] `.env` loader for API keys
- [x] `Config` object for runtime parameters
- [x] `LLMInterface` abstraction for multiple LLM backends
- [ ] PDF / EPUB ingestion
- [ ] Vector DB backend (SQLite + FAISS / Pinecone)
- [ ] Advanced multi-step planning & orchestration
- [ ] Multi-agent collaboration
- [ ] Hardened, stable scraping tool

---

## 📝 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 🤝 Contributing

Contributions are welcome! A few requests to keep the repo healthy:

- Run the build/tests locally before PRs
- Keep secrets out of commits (use `.env` and `.env.example`)
- Open issues for reproducible crashes and include `config.json` (sensitive values redacted)
- Keep changes scoped and add tests where appropriate

Thank you — happy hacking!

