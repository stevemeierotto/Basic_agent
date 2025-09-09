# basic_agent

`basic_agent` is a lightweight C++ framework for experimenting with AI agent design.  
It provides a modular structure for integrating memory, prompts, retrieval-augmented generation (RAG), and external tools.  
The project is built with **CMake** and is designed for extensibility, allowing you to add or swap out components like command processors, memory backends, or LLM interfaces.

---

## Features

- **Command Processor** – Handles input commands and dispatches them to the right modules.
- **Memory System** – Stores context and agent state in JSON (`memory.json`).
- **Prompt Factory** – Builds structured prompts for LLM interaction.
- **RAG (Retrieval-Augmented Generation)** – Allows knowledge retrieval from files or external sources.
- **Tooling Support** – Framework for integrating custom tools into the agent.
- **LLM Interface** – Abstraction layer to connect with any language model backend.

---

## Build Instructions

### Prerequisites
- A C++20 (or newer) compiler  
- [CMake](https://cmake.org/) (≥ 3.15 recommended)  

### Build Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/basic_agent.git
cd basic_agent

# Create and enter the build directory
mkdir build && cd build

# Configure the project
cmake ..

# Build
cmake --build .

Usage

After building, run the agent from the build/ directory.
It will load memory from memory.json and respond to commands through your chosen interface.

Example:

./basic_agent

You can customize the agent’s behavior by modifying:

    memory.json – Initial state and memory.

    prompt_factory.cpp – Prompt structures.

    tools.cpp – Extend with your own tools.

    rag.cpp – Retrieval logic for external data.

License

This project is licensed under the MIT License

.
You are free to use, modify, and distribute it under the terms of the license.
Roadmap / Ideas

    Add Python bindings for rapid prototyping.

    Extend RAG to support vector databases.

    Create plugins for different LLM providers.

    Implement more robust memory management.
