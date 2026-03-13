#include "../include/memory_repository_factory.h"
#include "../include/sqlite_memory_repository.h"
#include "../include/file_handler.h"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace Thoth {

std::unique_ptr<MemoryRepository> MemoryRepositoryFactory::createRepository(const Config& config, const std::string& preferredPath) {
    std::string sqlitePath = config.database_path;
    
    if (sqlitePath.empty()) {
        if (!preferredPath.empty()) {
            sqlitePath = preferredPath;
        } else {
            FileHandler fh;
            std::string baseWorkspace = fh.getAgentWorkspacePath();
            sqlitePath = (fs::path(baseWorkspace) / "memory.db").string();
        }
    }

    std::cout << "[MemoryRepositoryFactory] SQLite backend initialized (mandatory).\n";
    return std::make_unique<SQLiteMemoryRepository>(sqlitePath);
}

} // namespace Thoth
