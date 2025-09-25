#include "../include/chunkers/chunker.h"
#include <iostream>   // if you log warnings
#include <regex>      // if functions use regex
#include <sstream>    // if you split content by lines
#include <algorithm>  // if you clamp or transform
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

namespace Chunker {

std::vector<CodeChunk> createSmartChunks(const std::string& filePath,
                                                      const std::string& content) {
    std::vector<CodeChunk> chunks;
    auto ext = fs::path(filePath).extension().string();

    try {
        if (ext == ".md") {
            chunks = chunkByParagraphs(filePath, content);
        } else if (ext == ".txt") {
            // Prefer size-based chunking for novels/long text
            chunks = chunkBySize(filePath, content);
            if (chunks.empty()) {
                std::cerr << "[WARN] Size-based chunking produced no chunks for .txt file: "
                          << filePath << ". Falling back to paragraph-based.\n";
                chunks = chunkByParagraphs(filePath, content);
            }
        } else if (ext == ".cpp" || ext == ".h") {
            chunks = chunkByFunctions(filePath, content);
        } else {
            chunks = chunkBySize(filePath, content);
        }

        if (chunks.empty()) {
            std::cerr << "[ERROR] No chunks created for file: " << filePath
                      << ". Forcing single size-based chunk.\n";
            chunks = chunkBySize(filePath, content);
        }
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] Exception during chunking (" << filePath << "): "
                  << ex.what() << "\nFalling back to size-based chunking.\n";
        chunks = chunkBySize(filePath, content);
    }

    std::cerr << "[DEBUG] createSmartChunks produced " << chunks.size()
              << " chunk(s) for file: " << filePath << "\n";

    return chunks;
}

std::vector<CodeChunk> chunkByParagraphs(
    const std::string& filePath, const std::string& content)
{
    std::vector<CodeChunk> result;
    std::istringstream stream(content);
    std::string paragraph;
    std::string currentChunk;
    int lineNum = 1;
    int chunkStart = 1;

    while (std::getline(stream, paragraph)) {
        if (paragraph.empty()) {
            if (!currentChunk.empty()) {
                CodeChunk chunk;
                chunk.fileName = fs::absolute(filePath).string();
                chunk.symbolName = "";
                chunk.startLine = chunkStart;
                chunk.endLine = lineNum - 1;
                chunk.code = currentChunk;
                result.push_back(chunk);

                currentChunk.clear();
                chunkStart = lineNum + 1;
            }
        } else {
            currentChunk += paragraph + "\n";
        }
        lineNum++;
    }

    // Add final chunk if exists
    if (!currentChunk.empty()) {
        CodeChunk chunk;
        chunk.fileName = fs::absolute(filePath).string();
        chunk.symbolName = "";
        chunk.startLine = chunkStart;
        chunk.endLine = lineNum;
        chunk.code = currentChunk;
        result.push_back(chunk);
    }

    return result;
}

std::vector<CodeChunk> chunkByFunctions(
    const std::string& filePath, const std::string& content)
{
    std::vector<CodeChunk> result;
    std::istringstream stream(content);
    std::string line;
    std::string currentChunk;
    int lineNum = 1;
    int chunkStart = 1;

    std::regex functionRegex(R"(\s*([\w:~]+\s+)*[\w:~]+\s*\([^)]*\)\s*\{?)");

    while (std::getline(stream, line)) {
        currentChunk += line + "\n";

        if (std::regex_match(line, functionRegex) || (lineNum - chunkStart) > 50) {
            if (!currentChunk.empty()) {
                CodeChunk chunk;
                chunk.fileName = fs::absolute(filePath).string();
                chunk.symbolName = "";
                chunk.startLine = chunkStart;
                chunk.endLine = lineNum;
                chunk.code = currentChunk;
                result.push_back(chunk);

                currentChunk.clear();
                chunkStart = lineNum + 1;
            }
        }
        lineNum++;
    }

    if (!currentChunk.empty()) {
        CodeChunk chunk;
        chunk.fileName = fs::absolute(filePath).string();
        chunk.symbolName = "";
        chunk.startLine = chunkStart;
        chunk.endLine = lineNum;
        chunk.code = currentChunk;
        result.push_back(chunk);
    }

    return result;
}

std::vector<CodeChunk> chunkBySize(
    const std::string& filePath, const std::string& content)
{
    std::vector<CodeChunk> result;

    try {
        if (content.empty()) {
            std::cerr << "[WARN] chunkBySize called with empty content: "
                      << filePath << "\n";
            return result;
        }

        constexpr size_t CHUNK_SIZE = 4096;   // increased target chunk size (chars)
        constexpr size_t OVERLAP    = 512;    // sliding window overlap
        const std::string separators = ".!?"; // sentence boundaries

        size_t pos = 0;
        size_t totalSize = content.size();

        while (pos < totalSize) {
            size_t chunkEnd = pos + CHUNK_SIZE;
            if (chunkEnd > totalSize) chunkEnd = totalSize;

            // Extend chunkEnd to next sentence boundary for better context
            size_t sentencePos = chunkEnd;
            while (sentencePos < totalSize && separators.find(content[sentencePos]) == std::string::npos) {
                ++sentencePos;
            }
            if (sentencePos < totalSize) chunkEnd = sentencePos + 1; // include punctuation

            CodeChunk chunk;
            chunk.fileName = fs::absolute(filePath).lexically_normal().string();
            chunk.symbolName = "";
            chunk.startLine = 0; // not tracked for plain text
            chunk.endLine   = 0;
            chunk.code = content.substr(pos, chunkEnd - pos);

            // Remove null bytes
            chunk.code.erase(std::remove(chunk.code.begin(), chunk.code.end(), '\0'), chunk.code.end());

            // Skip chunks with too few word characters to avoid zero-norm embeddings
            size_t word_chars = std::count_if(chunk.code.begin(), chunk.code.end(),
                                              [](char c){ return std::isalnum(c); });
            if (word_chars < 20) { // increased threshold for better embedding quality
                std::cerr << "[WARN] Skipping low-content chunk at pos=" << pos
                          << " for file: " << filePath << "\n";
            } else {
                result.push_back(std::move(chunk));
            }

            if (chunkEnd >= totalSize) break; // reached end

            // advance with overlap
            if (chunkEnd < OVERLAP) pos = 0;
            else pos = chunkEnd - OVERLAP;
        }

        if (result.empty()) {
            std::cerr << "[ERROR] chunkBySize produced 0 valid chunks for file: "
                      << filePath << "\n";
        } else {
            std::cerr << "[DEBUG] chunkBySize produced " << result.size()
                      << " context-aware chunks for file: " << filePath << "\n";
        }

    } catch (const std::exception& ex) {
        std::cerr << "[EXCEPTION] chunkBySize failed for file: "
                  << filePath << " error=" << ex.what() << "\n";
    }

    return result;
}

} // namespace Chunker





