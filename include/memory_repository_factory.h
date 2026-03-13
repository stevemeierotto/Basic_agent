#pragma once
#include <memory>
#include <string>
#include "memory_repository.h"
#include "config.h"

namespace Thoth {

/**
 * @brief Factory for selecting and initializing the authoritative MemoryRepository.
 * Enforces single authoritative backend rules.
 */
class MemoryRepositoryFactory {
public:
    static std::unique_ptr<MemoryRepository> createRepository(const Config& config, const std::string& preferredPath = "");
};

} // namespace Thoth
