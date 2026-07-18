/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C2 Phase 3 / Plan M G3 chat prompt assembly constants
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CHAT_PROMPT_CONFIG_H
#define THOTH_CHAT_PROMPT_CONFIG_H

#include <cstddef>
#include <string>
#include <vector>

namespace Thoth {

namespace ChatPrompt {

/** Never truncated — instructs the model to stay within retrieved documents. */
inline constexpr const char* kGroundingRules =
    "Grounding Rules:\n"
    "- Answer using ONLY the [RAG Context] documents above.\n"
    "- Prefer quoting or paraphrasing the source; name the Document when helpful.\n"
    "- If the context does not contain the answer, say you do not have it in the indexed "
    "documents — do not guess or invent definitions.\n"
    "- Do not fabricate tool schemas, API fields, or acronyms not present in the context.\n";

/**
 * Plan M G3 — never truncated. Suppresses fake multi-turn / RAG transcript scripts.
 * Complements cue B (no open [Agent] slot).
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

/** Plan M G3 cue B — user turn prefix; full block is prefix + input + "\\n". */
inline constexpr const char* kUserTurnPrefix = "[User] ";

/**
 * Plan M G3 stop literals — retained for explicit / non-chat callers and tests.
 * Plan N N3: conversational chat does **not** send these via chatStopSequences().
 */
inline constexpr const char* kChatStopUser = "\n[User]";
inline constexpr const char* kChatStopAgent = "\n[Agent]";

inline std::string formatUserBlock(const std::string& user_input) {
    return std::string(kUserTurnPrefix) + user_input + "\n";
}

/** Plan N N3 / L6 — conversational chat sends an empty transcript stop list. */
inline std::vector<std::string> chatStopSequences() {
    return {};
}

} // namespace ChatPrompt

} // namespace Thoth

#endif // THOTH_CHAT_PROMPT_CONFIG_H
