#include "../include/command_processor.h"
#include "../include/file_handler.h"
#include "../include/webscraperTools.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <regex>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;


CommandProcessor::CommandProcessor(Memory& mem, RAGPipeline& rag, LLMInterface& llm)
    : memory(mem), rag(rag), llm(llm), promptFactory(mem, rag), scraper() 
{
    initializeCommands();
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

void CommandProcessor::handleCommand(const std::string& input) {
    if (!startsWith(input, "/")) {
        std::string response = processQuery(input);
        std::cout << "Assistant: " << response << "\n";
        memory.updateSummary(input, response);
        return;
    }

    auto [cmd, args] = parseCommand(input);
    
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
        "  /scrape             Scrape web and create summary\n"
        "  /rag                Query knowledge with RAG\n"
        "  /clear              Clears agent's memory and summaries\n"
        "  /backend ollama     Switch to Ollama\n"
        "  /backend openai     Switch to OpenAI\n"
        "Also: type 'exit' or 'quit' to leave.\n";
}

void CommandProcessor::clearMemory() {
    memory.clear();
    memory.save();
    std::cout << "Memory cleared.\n";
}

void CommandProcessor::ensureInitialized() {
    if (!initialized) {
        rag.init();
        FileHandler fh;
        rag.indexProject(fh.getRagDirectory());
        rag.saveIndex();
        initialized = true;
    }
}

void CommandProcessor::initializeCommands() {
    commandHandlers["help"] = [this](const std::string&) { showHelp(); };
    commandHandlers["h"] = commandHandlers["help"];
    commandHandlers["?"] = commandHandlers["help"];
    
    commandHandlers["clear"] = [this](const std::string&) { clearMemory(); };
    commandHandlers["reset"] = commandHandlers["clear"];
    
    commandHandlers["scrape"] = [this](const std::string& args) { handleScrape(args); };
    commandHandlers["rag"] = [this](const std::string& args) { handleRag(args); };
    commandHandlers["backend"] = [this](const std::string& args) { handleBackend(args); };
}

void CommandProcessor::handleScrape(const std::string& args) {
    if (args.empty()) {
        std::cout << "Usage: /scrape <search term>\n";
        return;
    }

    try {
        // Step 1: Perform search & summarization
        std::string summary = scraper.handleScrape(args, 3, 3);

        // Step 2: Fetch Reddit posts
        std::string redditRaw = scraper.fetchRedditPosts(args, 3);

        // Step 3: Summarize each Reddit post
        std::istringstream iss(redditRaw);
        std::string line;
        std::string redditSummary;
        while (std::getline(iss, line)) {
            if (!line.empty() && line.find("----") == std::string::npos) {
                redditSummary += scraper.summarizeText(line, 2) + " ";
            }
        }
        summary += "\n[Reddit Summary]\n" + redditSummary;

        // Step 4: Print the summary
        std::cout << "----- Summary -----\n" << summary << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error during scraping: " << e.what() << "\n";
    }
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
