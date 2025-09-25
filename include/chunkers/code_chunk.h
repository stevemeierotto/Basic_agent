#include <string>
#include <vector>


// Represents a chunk of code (function, class, or global block)
struct CodeChunk {
    std::string fileName;
    std::string symbolName; 
    int startLine;
    int endLine;
    std::string code;
    std::vector<float> embedding; // reserved for later
};
