#include "../include/memory.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;


Memory::Memory(const std::string& path) {
    if (!path.empty()) {
        filepath = path;
    } else {
        filepath = getDefaultPath();
    }

    // Only create parent directories if parent_path() is not empty
    fs::path parent = fs::path(filepath).parent_path();
    if (!parent.empty() && !fs::exists(parent)) {
        std::error_code ec;
        fs::create_directories(parent, ec);
        if (ec) {
            std::cerr << "[Memory] Failed to create directory: " << parent
                      << " (" << ec.message() << ")\n";
        }
    }

    load();
}



std::string Memory::getDefaultPath() const {
    namespace fs = std::filesystem;
#ifdef _WIN32
    char appdata[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(NULL, CSIDL_APPDATA, NULL, 0, appdata))) {
        fs::path base(appdata);
        return (base / "CodeAgentPlugin" / "memory.json").string();
    }
    return "memory.json"; // fallback
#else
    const char* home = getenv("HOME");
    fs::path base = home ? fs::path(home) : fs::current_path();
    fs::path dir = base / "code_agent_plugin";
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }
    return (dir / "memory.json").string();
#endif
}


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
            //save(); // write defaults
        }
    } else {
        // File does not exist: create directories if needed and save defaults
        namespace fs = std::filesystem;
        fs::path filePathObj(filepath);
        if (!fs::exists(filePathObj.parent_path())) {
            fs::create_directories(filePathObj.parent_path());
        }

        data = {
            {"conversation", json::array()},
            {"short_summary", ""},
            {"extended_summary", ""}
        };
        //save(); // write defaults
        std::cerr << "[Memory] Initialized new memory at " << filepath << "\n";
    }
}


void Memory::save() const {
    fs::path parent = fs::path(filepath).parent_path();
    if (!parent.empty() && !fs::exists(parent)) {
        std::error_code ec;
        fs::create_directories(parent, ec);
        if (ec) {
            std::cerr << "[Memory] Failed to create directory: " << parent
                      << " (" << ec.message() << ")\n";
            return;
        }
    }

    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "[Memory] Could not open file for writing: " << filepath << "\n";
        return;
    }
    out << data.dump(4);
}


void Memory::addMessage(const std::string& role, const std::string& content) {
    data["conversation"].push_back({
        {"role", role},
        {"content", content}
    });
   // save();
}

std::vector<json> Memory::getConversation() const {
    if (data.contains("conversation")) {
        return data["conversation"].get<std::vector<json>>();
    }
    return {};
}

void Memory::clear() {
    data = { 
        {"conversation", json::array()}, 
        {"short_summary", ""}, 
        {"extended_summary", ""} 
    };
    save();
}


void Memory::setSummary(const std::string& summary) {
    // For now, set the short summary directly
    data["short_summary"] = summary;
    //save();
}

std::string Memory::getSummary(bool useExtended /* = false */) const {
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


void Memory::updateSummary(const std::string& goal, const std::string& response) {
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

    save();

    // TODO: Later, replace the naive string truncation above with an LLM call
    // that summarizes the conversation more intelligently.
}

