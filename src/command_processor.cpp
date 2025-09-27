#include "command_processor.h"
#include "file_handler.h"
#include "index_manager.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <regex>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

CommandProcessor::CommandProcessor(Memory& mem, 
                                   RAGPipeline& ragPipeline, 
                                   LLMInterface& llmInterface,
                                   Config* cfg)  // added config pointer
    : memory(mem),
      rag(ragPipeline),
      llm(llmInterface),
      promptFactory(mem, ragPipeline),
      indexManager(ragPipeline.getIndexManager()) // pointer getter
{
    initializeCommands();
    
}

void CommandProcessor::handleSimilarityCommand(const std::string& args) {
    // Available options
    std::unordered_map<std::string, std::unique_ptr<ISimilarity>> options;
    options["dot"] = std::make_unique<DotProductSimilarity>();
    options["cosine"] = std::make_unique<CosineSimilarity>();
    options["euclidean"] = std::make_unique<EuclideanSimilarity>();
    options["jaccard"] = std::make_unique<JaccardSimilarity>();

    std::string chosen = toLower(trim(args));

    // If no argument passed, interactively prompt
    if (chosen.empty()) {
        std::cout << "Available similarity methods:\n";
        for (auto& [name, _] : options) std::cout << "  " << name << "\n";
        std::cout << "Enter choice: ";
        std::getline(std::cin, chosen);
        chosen = toLower(trim(chosen));
    }

    auto it = options.find(chosen);
    if (it == options.end()) {
        std::cout << "Unknown similarity: " << chosen << "\n";
        return;
    }

    // Apply the chosen similarity
    rag.getIndexManager()->store.setSimilarity(std::move(it->second));
    std::cout << "Similarity set to " << chosen << "\n";
}


std::string CommandProcessor::processQuery(const std::string& input) {
    ensureInitialized();
    // 1. Retrieve relevant context from RAG
    std::vector<CodeChunk> contextChunks = rag.retrieveRelevant(input, {}, 5); // top 5
    std::ostringstream contextStream;
    for (auto& c : contextChunks) {
        contextStream << c.code << "\n---\n";
    }
    std::string ragContext = contextStream.str();

    // 2. Build a conversation prompt including RAG context
    std::string convPrompt = promptFactory.buildConversationPrompt(input);
    std::string finalPrompt;
    if (!ragContext.empty()) {
        finalPrompt = "[RAG Context]\n" + ragContext + "\n[User Query]\n" + convPrompt;
    } else {
        finalPrompt = convPrompt;
    }
    std::cout << "FINALPROMPT!!!!! " << finalPrompt << "\n";
    std::cout << "[ProcessQuery] Prompt created with RAG context, waiting for response.\n";

    // 3. Query the LLM
    std::string response = llm.query(finalPrompt);

    // 4. Update memory with error handling
    try {
        memory.addMessage("user", input);
        memory.addMessage("assistant", response);
        memory.save();
        
        // Only update summary if everything succeeded
        memory.updateSummary(input, response);
    } catch (const std::exception& e) {
        std::cerr << "[WARNING] Failed to save to memory: " << e.what() << std::endl;
        // Don't fail the entire operation
    }

    return response;
}

std::string CommandProcessor::trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::string CommandProcessor::lstripSlash(const std::string& s) {
    if (!s.empty() && s[0] == '/') return s.substr(1);
    return s;
}

std::string CommandProcessor::toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

bool CommandProcessor::startsWith(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

void CommandProcessor::runLoop() {
    std::cout << "Basic Chat Agent . Type /help for commands. Type exit or quit to leave.\n";
    std::string line;

    try {
        ensureInitialized();
        std::cout << "RAG system ready.\n";
    } catch (const std::exception& e) {
        std::cout << "Warning: RAG system initialization failed: " << e.what() << "\n";
        std::cout << "Some features may be limited.\n";
    }

    while (true) {
        std::cout << "<USER> " << std::flush;
        if (!std::getline(std::cin, line)) {
            std::cout << "\nEOF received. Exiting.\n";
            break;
        }

        line = trim(line);
        if (line.empty()) continue;

        // exit conditions (with or without slash)
        std::string low = toLower(line);
        if (low == "exit" || low == "quit" || low == "/exit" || low == "/quit") {
            std::cout << "Goodbye.\n";
            break;
        }
        std::cout << std::endl;
        handleCommand(line);

    }
}

void CommandProcessor::showConfig() const {
    if(config) config->printConfig();
    else std::cout << "No config connected.\n";
}

void CommandProcessor::setConfig(const std::string& key, const std::string& value) {
    if (config) {
        if (config->set(key, value)) {
            std::cout << "Updated " << key << " to " << value << "\n";
        } else {
            std::cout << "Failed to update key: " << key << "\n";
        }
    } else {
        std::cout << "No config connected.\n";
    }
}


void CommandProcessor::handleCommand(const std::string& input) {
    if (!startsWith(input, "/")) {
        std::string response = processQuery(input);
        std::cout << "Assistant: " << response << "\n";
        memory.updateSummary(input, response);
        return;
    }

    auto [cmd, args] = parseCommand(input);

    // Handle Config commands first
    if (cmd == "show" && args == "config") {
        if (config) config->printConfig();
        else std::cout << "No config connected.\n";
        return;
    }

    if (cmd == "set") {
        std::istringstream iss(args);
        std::string key, value;
        iss >> key >> value;
        if (key.empty() || value.empty()) {
            std::cout << "Usage: /set <key> <value>\n";
        } else if (config) {
            if(config->set(key, value))
                std::cout << "Updated " << key << " to " << value << "\n";
            else
                std::cout << "Failed to update key: " << key << "\n";
        } else {
            std::cout << "No config connected.\n";
        }
        return;
    }

    // Fallback to existing command handlers
    auto it = commandHandlers.find(cmd);
    if (it != commandHandlers.end()) {
        try {
            it->second(args);
        } catch (const std::exception& e) {
            std::cout << "Error executing command: " << e.what() << "\n";
        }
    } else {
        std::cout << "Unknown command '/" << cmd << "'. Try /help.\n";
    }
}


std::pair<std::string, std::string> CommandProcessor::parseCommand(const std::string& input) {
    std::string stripped = input;
    if (!stripped.empty() && stripped[0] == '/')
        stripped.erase(0, 1);  // remove leading slash

    std::istringstream iss(stripped);
    std::string cmd;
    std::getline(iss, cmd, ' ');
    std::string args;
    std::getline(iss, args);
    return { toLower(cmd), trim(args) };  // assumes trim() and toLower() exist
}

void CommandProcessor::showHelp() {
    std::cout <<
        "Built-ins:\n"
        "  /help               Show this help\n"
        "  /rag                Query knowledge with RAG\n"
        "  /clear              Clears agent's memory and summaries\n"
        "  /backend ollama     Switch to Ollama\n"
        "  /backend openai     Switch to OpenAI\n"
        "  /similarity         Switch Similarity\n"
        "  /config             Show config values"
        "  /set temerature     0.5 etc less than 1\n"
        "Also: type 'exit' or 'quit' to leave.\n";
}

void CommandProcessor::clearMemory() {
    memory.clear();
    memory.save();
    std::cout << "Memory cleared.\n";
}

void CommandProcessor::ensureInitialized() {
    if (!initialized) {
        FileHandler fh;
        indexManager->init(fh.getRagDirectory());
        
        indexManager->indexProject(fh.getRagDirectory());
        indexManager->saveIndex();
        initialized = true;
    }
}

void CommandProcessor::initializeCommands() {
    commandHandlers["help"] = [this](const std::string&) { showHelp(); };
    commandHandlers["h"] = commandHandlers["help"];
    commandHandlers["?"] = commandHandlers["help"];
    
    commandHandlers["clear"] = [this](const std::string&) { clearMemory(); };
    commandHandlers["reset"] = commandHandlers["clear"];
    commandHandlers["rag"] = [this](const std::string& args) { handleRag(args); };
    commandHandlers["backend"] = [this](const std::string& args) { handleBackend(args); };
    commandHandlers["similarity"] = [this](const std::string& args) {
    handleSimilarityCommand(args); };
    commandHandlers["config"] = [this](const std::string&) { showConfig(); };


}

void CommandProcessor::handleRag(const std::string& args) {
    if (args.empty()) {
        std::cout << "Usage: /rag <your query>\n";
        return;
    }

    auto chunks = rag.retrieveRelevant(args, {}, 5); // top 5
    if (chunks.empty()) {
        std::cout << "[RAG] No relevant context found.\n";
        return;
    }

    std::cout << "[RAG] Top relevant chunks:\n";
    int idx = 1;
    for (const auto& c : chunks) {
        std::cout << "Chunk " << idx++ << " (" << c.fileName
                  << " lines " << c.startLine << "-" << c.endLine << "):\n";
        std::cout << c.code << "\n---\n";
    }
}

void CommandProcessor::handleBackend(const std::string& args) {
    if (args == "ollama") {
        llm.setBackend(LLMBackend::Ollama);
        std::cout << "Switched backend to Ollama\n";
    } else if (args == "openai") {
        llm.setBackend(LLMBackend::OpenAI);
        std::cout << "Switched backend to OpenAI\n";
    } else {
        std::cout << "Usage: /backend [ollama|openai]\n";
    }
}
