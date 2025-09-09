#include "../include/command_processor.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <regex>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;


CommandProcessor::CommandProcessor(Memory& mem, RAGPipeline& rag, LLMInterface& llm)
    : memory(mem), rag(rag), llm(llm), promptFactory(mem, rag) {}

std::string CommandProcessor::processQuery(const std::string& input) {
    // 1. Build a conversation + RAG prompt
    std::cout << "[ProcessQuery] we begin! \n";
    std::string prompt = promptFactory.buildConversationPrompt(input);
    std::cout << "[ProcessQuery] Prompt created , waiting for response. \n";
    // 2. Query the LLM
    std::string response = llm.query(prompt);
    std::cout << "[ProcessQuery] response recieved. \n";

    // 3. Update memory
    memory.addMessage("user", input);
    std::cout << "[ProcessQuery] addMessage user \n";
    memory.addMessage("assistant", response);
    std::cout << "[ProcessQuery] addmessage assistant \n";
    memory.save();
    std::cout << "[ProcessQuery] memory saved return response \n";
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



