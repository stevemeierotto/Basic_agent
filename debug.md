# Apollo Debug Log

## Project Overview
Apollo (basic_agent) is an AI-powered agent written in C++20 that provides:
- **Memory System**: Persistent JSON-based long-term memory for conversations and summaries
- **RAG (Retrieval-Augmented Generation)**: Embedding-based search across documents and code using TF-IDF or other embedding methods
- **LLM Integration**: Abstraction layer supporting Ollama (local) and OpenAI (API) backends
- **Modular Tools**: Web search, summarization, command execution (some experimental)
- **Vector Store**: Pluggable similarity metrics (cosine, dot-product, euclidean, jaccard)

The agent can run in two modes:
1. **Standalone CLI**: Direct executable (`basic_agent_cli`) with interactive REPL
2. **Plugin Mode**: Embedded as `BasicAgentPlugin` in the Thoth control panel GUI application

Key dependencies:
- **C++20** compiler (GCC 10+, Clang 12+, MSVC 2019+)
- **CMake 3.15+** for build system
- **libcurl** for HTTP/HTTPS requests to LLM APIs
- **nlohmann/json** (header-only) for JSON parsing
- **wxWidgets** (only for GUI integration, not for standalone agent)

## How Apollo Starts

### Standalone Mode
**Entry Point**: `main.cpp` (lines 10-36)

**Startup Sequence**:
1. Loads `config.json` from current working directory via `Config::loadFromJson()`
2. Attempts to load `.env` file from parent directory (`../.env`) via `EnvLoader::loadEnvFile()`
   - Falls back to system environment variables if file not found
3. Initializes core components:
   - `Memory` object (defaults to `agent_workspace/memory.json`)
   - `LLMInterface` with Ollama backend (default)
   - `EmbeddingEngine` with TF-IDF method (default)
   - `IndexManager` (manages code chunks and indexing)
   - `RAGPipeline` (owns the embedding engine)
   - `CommandProcessor` (handles REPL loop)
4. Calls `CommandProcessor::runLoop()` which starts an interactive command loop

**Command to Launch**:
```bash
cd external/basic_agent/build
./basic_agent_cli
```

### Plugin Mode
**Entry Point**: `BasicAgentPlugin` constructor (`src/basic_agent_plugin.cpp`, lines 8-50)

**Startup Sequence**:
1. Creates `EmbeddingEngine` with TF-IDF method
2. Creates `IndexManager` (raw pointer, owned by plugin)
3. Creates `RAGPipeline` (takes ownership of embedding engine)
4. Creates `LLMInterface` with Ollama backend
5. Creates `CommandProcessor` (references memory, RAG, LLM)
6. Loads `config.json` from current working directory
7. Attempts to load `.env` from parent directory (`../.env`)
8. Loads or creates RAG index at `agent_workspace/rag/rag_index.bin`

**Integration**: Used by `AgentInterface` class in `src/AgentInterface.cpp` which wraps the plugin for the wxWidgets GUI application.

**Command to Launch** (GUI mode):
```bash
cd build
./thoth-control-panel
```

### Configuration Files
- **`config.json`**: Runtime parameters (temperature, top_p, max_tokens, similarity_threshold, verbosity)
- **`.env`**: API keys (OPENAI_API_KEY, GOOGLE_CSE_ID, GOOGLE_API_KEY) - loaded from parent directory in both modes
- **`agent_workspace/memory.json`**: Persistent conversation memory
- **`agent_workspace/rag/rag_index.bin`**: Vector store index for RAG

## Current Architecture Notes

### Component Hierarchy
```
BasicAgentPlugin (or main.cpp)
├── Config (runtime parameters)
├── Memory (persistent JSON storage)
├── EmbeddingEngine (TF-IDF, Simple, WordHash, External)
├── IndexManager (owns VectorStore, manages CodeChunks)
├── RAGPipeline (owns EmbeddingEngine, uses IndexManager)
├── LLMInterface (Ollama/OpenAI via libcurl)
└── CommandProcessor (REPL loop, command routing)
    └── PromptFactory (builds prompts from Memory + RAG context)
```

### Data Flow for Query Processing
1. **User Input** → `CommandProcessor::handleCommand()` or `processQuery()`
2. **Command Detection**: If input starts with `/`, routes to command handlers; otherwise treats as query
3. **RAG Retrieval**: `RAGPipeline::retrieveRelevant()` → `IndexManager::retrieveChunks()` → `VectorStore::retrieve()`
   - Embeds query using `EmbeddingEngine`
   - Computes similarity scores using pluggable `ISimilarity` implementations
   - Returns top-K relevant `CodeChunk` objects
4. **Prompt Building**: `PromptFactory::buildConversationPrompt()` combines:
   - Recent conversation history from `Memory`
   - RAG context chunks
   - User query
5. **LLM Query**: `LLMInterface::query()` sends HTTP request to Ollama or OpenAI
6. **Memory Update**: Response saved to `Memory` with auto-save (dirty flag + 5-minute interval)
7. **Summary Update**: `Memory::updateSummary()` condenses conversation

### Key Design Patterns
- **Ownership**: `RAGPipeline` owns `EmbeddingEngine` via `unique_ptr`; `IndexManager` uses raw pointer to engine
- **Thread Safety**: `Memory` uses `std::mutex`; `IndexManager` uses `std::shared_mutex` for chunks
- **Pluggable Similarity**: `VectorStore` accepts `std::unique_ptr<ISimilarity>` for different metrics
- **Command Pattern**: `CommandProcessor` uses `std::unordered_map<std::string, CommandHandler>` for extensible commands

### File Structure
- **`include/`**: Header files (17 headers)
- **`src/`**: Implementation files (15 .cpp files)
- **`agent_workspace/`**: Runtime data (memory.json, rag/ directory)
- **`build/`**: CMake build artifacts and compiled library (`libbasic_agent.so`)

## Known Issues (Initial)

### Path Hardcoding
- **Location**: `main.cpp:16`, `basic_agent_plugin.cpp:24`
- **Issue**: Hardcoded `"../.env"` path assumes parent directory structure
- **Impact**: May fail if run from different working directories
- **Note**: Should use relative path from executable or configurable path

### Debug Output in Production Code
- **Location**: `command_processor.cpp:78`
- **Issue**: `std::cout << "FINALPROMPT!!!!! " << finalPrompt << "\n";` appears to be debug code left in
- **Impact**: Verbose output in production, may expose internal prompt structure
- **Note**: Should be conditional on verbosity level or removed

### Memory Management Concern
- **Location**: `basic_agent_plugin.cpp:10, 54`
- **Issue**: `IndexManager* indexManager` is created with `new` and deleted in destructor
- **Impact**: Raw pointer ownership, potential for leaks if exception thrown
- **Note**: Could use `unique_ptr` for safer ownership

### Error Handling Gaps
- **Location**: Multiple files
- **Issue**: Some operations lack try-catch blocks (e.g., file I/O, network requests)
- **Impact**: Unhandled exceptions could crash the agent
- **Note**: `command_processor.cpp:92-95` shows good pattern with try-catch around memory operations

### Thread Safety in VectorStore
- **Location**: `vector_store.h:21-30`
- **Issue**: `embeddings` and `documents` vectors are public and mutable without explicit locking
- **Impact**: Potential race conditions in multi-threaded scenarios
- **Note**: `IndexManager` has `chunksMutex` but `VectorStore` itself doesn't

### Inconsistent Error Messages
- **Location**: Various files
- **Issue**: Some errors use `std::cerr`, others use `std::cout`, some return error strings
- **Impact**: Inconsistent logging makes debugging harder
- **Note**: Plugin mode uses `std::cerr` consistently, but standalone mode mixes streams

### Configuration File Path Assumptions
- **Location**: `main.cpp:13`, `basic_agent_plugin.cpp:16`
- **Issue**: Assumes `config.json` exists in current working directory
- **Impact**: Fails silently or uses defaults if file missing
- **Note**: Both modes check existence but behavior differs slightly

### RAG Index Path Logic
- **Location**: `basic_agent_plugin.cpp:31`
- **Issue**: Uses `FileHandler::getRagPath()` which may have path resolution logic
- **Impact**: Unclear if path resolution is consistent across different execution contexts
- **Note**: Should verify path resolution works in both standalone and plugin modes

### Missing Initialization Check
- **Location**: `command_processor.cpp:61` (`ensureInitialized()`)
- **Issue**: `initialized` flag exists but initialization logic not fully visible in reviewed code
- **Impact**: May have race conditions or incomplete initialization
- **Note**: Need to review full `ensureInitialized()` implementation

## Open Questions

1. **Plugin Interface**: `basic_agent_plugin.h:14` has commented out `#include "Plugin.h"` - is there a planned plugin interface system, or is this legacy code?

2. **Tools Implementation**: `tools.h` defines a virtual base class `Tools` with only `summarizeText()`, but README mentions web search, command execution tools. Where are these implemented?

3. **Web Scraping**: README mentions experimental web scraping that can cause faults. Where is this code located? (Not found in standard file structure)

4. **External Embedding**: `EmbeddingEngine::Method::External` exists but implementation unclear - how does it integrate with external embedding services?

5. **Memory Limits**: `Config` has `memory_limit_mb` and `disk_quota_mb` but unclear if these are enforced or just documentation

6. **RAG Index Format**: Index is saved as `.bin` file but format/schema not clear from headers - is it binary JSON, custom format, or something else?

7. **Command Processor Threading**: `AgentInterface::processUserInput()` spawns a detached thread for async processing, but `CommandProcessor` appears designed for synchronous REPL. Are there thread safety concerns when used in plugin mode?

8. **Similarity Threshold**: `VectorStore` has hardcoded `SIMILARITY_THRESHOLD = 0.01f` but `Config` has `similarity_threshold` - which one is actually used?

9. **Chunk Size Limits**: `IndexManager` has constants like `MAX_CHUNK_SIZE = 4096` and `MAX_CHUNKS = 10000` - are these enforced, and what happens when limits are reached?

10. **Environment Variable Fallback**: If `.env` file is missing, system environment variables are used, but it's unclear which variables are checked and in what order.

## Recent Debug Entries
(Leave this section empty for now — we will append to it over time.)
