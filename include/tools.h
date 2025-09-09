/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * basic_agent - AI Agent with Memory and RAG Capabilities
 * uses either Ollama lacal models or OpenAI API
 *
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text
 */

#pragma once
#include <string>

namespace Tools {

    // Replace all occurrences of oldName with newName in file.
    bool replaceName(const std::string& filePath,
                     const std::string& oldName,
                     const std::string& newName);

}

