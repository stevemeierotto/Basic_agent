/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C2 Phase 3 / Plan M G3 chat prompt assembly constants
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CHAT_PROMPT_CONFIG_H
#define THOTH_CHAT_PROMPT_CONFIG_H

#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

namespace Thoth {

namespace ChatPrompt {

/** Never truncated — instructs the model to stay within retrieved documents. */
inline constexpr const char* kGroundingRules =
    "Grounding Rules:\n"
    "- Answer using ONLY the [RAG Context] documents above.\n"
    "- Use the retrieved passages as internal reference material; answer in normal prose.\n"
    "- Paraphrase or quote source content when helpful; mention a filename naturally only when "
    "it helps the user.\n"
    "- Never reproduce internal retrieval formatting (Document:, source_span=, or --- separators).\n"
    "- If the context does not contain the answer, say you do not have it in the indexed "
    "documents — do not guess or invent definitions.\n"
    "- Do not fabricate tool schemas, API fields, or acronyms not present in the context.\n";

/**
 * CSG-B — never truncated. Suppresses echo of RAG chunk injection scaffolding.
 * Complements kGroundingRules and kAntiTranscriptRules.
 */
inline constexpr const char* kAntiRegurgitationRules =
    "Retrieval Format Rules:\n"
    "- The [RAG Context] block uses internal Document:/source_span= labels for grounding only.\n"
    "- Your reply must be plain assistant prose — never echo those labels or chunk separators.\n";

/** Appended once on CSG-B regurgitation retry. */
inline constexpr const char* kRegurgitationRetryReminder =
    "\n\nReminder: Answer in plain prose only. Do not copy Document:, source_span=, or --- "
    "formatting from the context.\n";

/**
 * Plan M G3 — never truncated. Suppresses fake multi-turn / RAG transcript scripts.
 * Complements explicit assistant completion cue after the user block.
 */
inline constexpr const char* kAntiTranscriptRules =
    "Response Rules:\n"
    "- Give one direct assistant answer only.\n"
    "- Do not invent further [User] or [Agent] turns.\n"
    "- Do not invent [RAG Context], transcript scaffolding, or chat-log formatting.\n";

inline constexpr const char* kRagContextHeader = "[RAG Context]\n";
inline constexpr const char* kUserQueryHeader = "[User Query]\n";

inline constexpr std::size_t kMaxSystemPromptChars = 1024;

/** Plan M G3 — explicit chat-path generation ceiling (matches Config::max_tokens default). */
inline constexpr int kChatMaxTokens = 512;

/** User turn prefix; paired with kAgentTurnPrefix for completion boundary. */
inline constexpr const char* kUserTurnPrefix = "[User] ";

/** Explicit assistant slot — generation continues after this prefix (Phase 3 boundary). */
inline constexpr const char* kAgentTurnPrefix = "[Agent] ";

/**
 * Plan M G3 user block + Phase 3 assistant completion cue (ungrounded chat path).
 * Full boundary: "[User] " + input + "\\n" + "[Agent] "
 */
inline std::string formatUserBlock(const std::string& user_input) {
    return std::string(kUserTurnPrefix) + user_input + "\n" + kAgentTurnPrefix;
}

/** Grounded chat path — query lives under [User Query]; only the assistant cue remains. */
inline std::string formatAgentCompletionCue() {
    return std::string(kAgentTurnPrefix);
}

/**
 * Plan M G3 stop literals — retained for explicit / non-chat callers and tests.
 * Plan N N3: conversational chat does **not** send these via chatStopSequences().
 */
inline constexpr const char* kChatStopUser = "\n[User]";
inline constexpr const char* kChatStopAgent = "\n[Agent]";

/** Phase A chat migration — chat (default) vs legacy /v1/completions rollback. */
enum class ChatInferenceMode {
    Completions,
    Chat,
};

/** Experiment A.1 — Qwen ChatML turn boundaries (chat inference path only). */
inline constexpr const char* kChatMlStopImEnd = "<|im_end|>";
inline constexpr const char* kChatMlStopImStart = "<|im_start|>";

/**
 * Plan N N3 / L6 — completions path sends an empty transcript stop list.
 * Experiment A.1 — chat inference adds ChatML turn-boundary stops only.
 */
inline std::vector<std::string> chatStopSequences(
    ChatInferenceMode mode = ChatInferenceMode::Completions) {
    if (mode == ChatInferenceMode::Chat) {
        return {kChatMlStopImEnd, kChatMlStopImStart};
    }
    return {};
}

inline ChatInferenceMode chatInferenceModeFromEnv() {
    const char* env = std::getenv("THOTH_CHAT_INFERENCE_MODE");
    if (env == nullptr || env[0] == '\0') {
        return ChatInferenceMode::Chat;
    }
    std::string value(env);
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (value == "chat") {
        return ChatInferenceMode::Chat;
    }
    return ChatInferenceMode::Completions;
}

inline const char* chatInferenceModeLabel(ChatInferenceMode mode) {
    switch (mode) {
    case ChatInferenceMode::Chat:
        return "chat";
    case ChatInferenceMode::Completions:
    default:
        return "completions";
    }
}

} // namespace ChatPrompt

} // namespace Thoth

#endif // THOTH_CHAT_PROMPT_CONFIG_H
