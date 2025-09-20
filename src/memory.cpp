#include "../include/memory.h"
#include "../include/file_handler.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

Memory::~Memory() {
    flush();  // ensure all data saved
}
// ------------------------
// Constructor
// ------------------------
Memory::Memory(const std::string& path) {
    if (!path.empty()) {
        filepath = path;
    } else {
        FileHandler fh;
        filepath = fh.getMemoryPath();
    }

    // Create parent directories if needed
    fs::path parent = fs::path(filepath).parent_path();
    if (!parent.empty() && !fs::exists(parent)) {
        std::error_code ec;
        fs::create_directories(parent, ec);
        if (ec) {
            std::cerr << "[Memory] Failed to create directory: " << parent
                      << " (" << ec.message() << ")\n";
        }
    }

    lastSave = std::chrono::steady_clock::now();
    load();
}

// ------------------------
// Conversation
// ------------------------
void Memory::addMessage(const std::string& role, const std::string& content) {
    std::lock_guard<std::mutex> lock(mtx);
    data["conversation"].push_back({
        {"role", role},
        {"content", content},
        {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()}
    });
    markDirty();

    // Auto-save every 10 messages or interval exceeded
    if (data["conversation"].size() % 10 == 0) {
        saveIfNeeded();
    }
}

void Memory::addMessages(const std::vector<std::pair<std::string, std::string>>& messages) {
    std::lock_guard<std::mutex> lock(mtx);
    for (const auto& [role, content] : messages) {
        data["conversation"].push_back({
            {"role", role},
            {"content", content},
            {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()}
        });
    }
    markDirty();
}

// ------------------------
// Flush / Save
// ------------------------
void Memory::flush() const {
    std::lock_guard<std::mutex> lock(mtx);
    saveUnlocked();
}

void Memory::save() const {
    std::lock_guard<std::mutex> lock(mtx);
    saveIfNeeded();
}

// ------------------------
// Private helpers
// ------------------------
void Memory::markDirty() const {
    isDirty = true;
}

void Memory::saveIfNeeded() const {
    auto now = std::chrono::steady_clock::now();
    if (isDirty && (now - lastSave) > AUTO_SAVE_INTERVAL) {
        saveUnlocked();
    }
}

void Memory::saveUnlocked() const {
    try {
        std::ofstream out(filepath);
        if (out) {
            out << data.dump(4);  // pretty JSON
            isDirty = false;
            lastSave = std::chrono::steady_clock::now();
        } else {
            std::cerr << "[Memory] Failed to open file for saving: " << filepath << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[Memory] Exception during save: " << e.what() << "\n";
    }
}
//below is the original section before editting
// ------------------------
// Load memory from file
// ------------------------
void Memory::load() {
    std::ifstream in(filepath);
    if (in) {
        try {
            in >> data;

            // Ensure required fields exist
            if (!data.contains("conversation"))
                data["conversation"] = json::array();
            if (!data.contains("short_summary"))
                data["short_summary"] = "";
            if (!data.contains("extended_summary"))
                data["extended_summary"] = "";

            std::cerr << "[Memory] Loaded memory from " << filepath << "\n";
        } catch (...) {
            std::cerr << "[Memory] Failed to parse JSON, reinitializing.\n";
            data = {
                {"conversation", json::array()},
                {"short_summary", ""},
                {"extended_summary", ""}
            };
        }
    } else {
        // File does not exist: create directories if needed and initialize defaults
        fs::path filePathObj(filepath);
        if (!fs::exists(filePathObj.parent_path())) {
            fs::create_directories(filePathObj.parent_path());
        }

        data = {
            {"conversation", json::array()},
            {"short_summary", ""},
            {"extended_summary", ""}
        };
    }
}

// ------------------------
// Get conversation messages
// ------------------------
std::vector<json> Memory::getConversation() const {
    std::lock_guard<std::mutex> lock(mtx);
    if (data.contains("conversation")) {
        return data["conversation"].get<std::vector<json>>();
    }
    return {};
}

// ------------------------
// Clear memory
// ------------------------
void Memory::clear() {
    std::lock_guard<std::mutex> lock(mtx);
    data = { 
        {"conversation", json::array()}, 
        {"short_summary", ""}, 
        {"extended_summary", ""} 
    };
    markDirty();
    saveUnlocked();
}



// ------------------------
// Set short summary
// ------------------------
void Memory::setSummary(const std::string& summary) {
    std::lock_guard<std::mutex> lock(mtx);
    data["short_summary"] = summary;
    markDirty();
    saveIfNeeded();
}

// ------------------------
// Get summary
// ------------------------
std::string Memory::getSummary(bool useExtended /*= false*/) const {
    std::lock_guard<std::mutex> lock(mtx);
    try {
        if (useExtended && data.contains("extended_summary") && data["extended_summary"].is_string()) {
            return data["extended_summary"].get<std::string>();
        }
        if (data.contains("short_summary") && data["short_summary"].is_string()) {
            return data["short_summary"].get<std::string>();
        }
    } catch (...) {
        std::cerr << "[Memory] Error while reading summary.\n";
    }
    return "[Memory] No summary available.";
}

// ------------------------
// Update short & extended summaries
// ------------------------
void Memory::updateSummary(const std::string& goal, const std::string& response) {
    std::lock_guard<std::mutex> lock(mtx);

    // --- Short summary (always replaced) ---
    std::ostringstream shortOss;
    shortOss << "Last Goal: " << goal << "\n";
    shortOss << "Last Response: " << response.substr(0, 200) << "...";
    data["short_summary"] = shortOss.str();

    // --- Extended summary (append with cap) ---
    if (!data.contains("extended_summary") || !data["extended_summary"].is_string()) {
        data["extended_summary"] = "";
    }

    std::string extended = data["extended_summary"].get<std::string>();
    extended += "\n- Goal: " + goal + " | Resp: " + response.substr(0, 120) + "...";

    // Cap extended summary size (keep only last ~5000 chars)
    constexpr size_t MAX_EXTENDED_SIZE = 5000;
    if (extended.size() > MAX_EXTENDED_SIZE) {
        extended = extended.substr(extended.size() - MAX_EXTENDED_SIZE);
    }

    data["extended_summary"] = extended;

    markDirty();
    saveIfNeeded();
}


