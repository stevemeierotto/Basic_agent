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
    : memory(mem), rag(rag), llm(llm), promptFactory(mem, rag), scraper() // uses default constructor
{
    rag.init();
    FileHandler fh;
    rag.indexProject(fh.getRagDirectory());
    rag.saveIndex();
}


std::string CommandProcessor::processQuery(const std::string& input) {
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

    std::cout << "[ProcessQuery] Prompt created with RAG context, waiting for response.\n";

    // 3. Query the LLM
    std::string response = llm.query(finalPrompt);

    // 4. Update memory
    memory.addMessage("user", input);
    memory.addMessage("assistant", response);
    memory.save();

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
        // Default path → send through memory + RAG + LLM
        std::string response = processQuery(input);
        std::cout << "Assistant: " << response << "\n";

        // Update summaries after each exchange
        memory.updateSummary(input, response);
        return;
    }

    std::string stripped = lstripSlash(input);
    std::istringstream iss(stripped);
    std::string cmd;
    std::getline(iss, cmd, ' ');
    std::string args;
    std::getline(iss, args);
    cmd = toLower(cmd);
    args = trim(args);

    if (cmd == "help" || cmd == "h" || cmd == "?") {
        std::cout <<
            "Built-ins:\n"
            "  /help               Show this help\n"
            "  /scrape             Scrape web and create summery."
            "  /rag                Query knowledge with RAG\n"
            "  /clear              Clears agent's memory and summaries\n"
            "  /backend ollama     Switch to Ollama\n"
            "  /backend openai     Switch to OpenAI\n"
            "Also: type 'exit' or 'quit' to leave.\n";
        return;
    }

    if (cmd == "clear" || cmd == "reset") {
        memory.clear();
        memory.save();
        std::cout << "Memory cleared.\n";
        return;
    }

if (cmd == "scrape") {
    if (args.empty()) {
        std::cout << "Usage: /scrape <search term>\n";
    } else {
        try {
            // Perform the search and summarization
            std::string summary = scraper.handleScrape(args, 3, 3);
            std::cout << "Fetching Reddit posts.\n";
            // Fetch Reddit posts using the user’s search query
            std::string redditRaw = scraper.fetchRedditPosts(args, 3);

            // Summarize each post individually
            std::istringstream iss(redditRaw);
            std::string line;
            std::string redditSummary;
            while (std::getline(iss, line, '\n')) {
                if (!line.empty() && line.find("----") == std::string::npos) {
                    redditSummary += scraper.summarizeText(line, 2) + " "; // summarize 2 sentences per line
                }
            }
            summary += "\n[Reddit Summary]\n" + redditSummary;
            // Print the summary
            std::cout << "----- Summary -----\n" << summary << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
        }
    }
    return;  // Prevent fall-through to "Unknown command"
}


    if (cmd == "rag") {
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
        for (auto& c : chunks) {
            std::cout << "Chunk " << idx++ << " (" << c.fileName
                      << " lines " << c.startLine << "-" << c.endLine << "):\n";
            std::cout << c.code << "\n---\n";
        }
        return;
    }

    if (cmd == "backend") {
        if (args == "ollama") {
            llm.setBackend(LLMBackend::Ollama);
            std::cout << "Switched backend to Ollama\n";
        } else if (args == "openai") {
            llm.setBackend(LLMBackend::OpenAI);
            std::cout << "Switched backend to OpenAI\n";
        } else {
            std::cout << "Usage: /backend [ollama|openai]\n";
        }
        return;
    }

    std::cout << "Unknown command '/" << cmd << "'. Try /help.\n";
}



