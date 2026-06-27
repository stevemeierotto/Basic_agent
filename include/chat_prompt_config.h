/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C2 Phase 3 chat prompt assembly constants
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CHAT_PROMPT_CONFIG_H
#define THOTH_CHAT_PROMPT_CONFIG_H

#include <cstddef>

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

inline constexpr const char* kRagContextHeader = "[RAG Context]\n";
inline constexpr const char* kUserQueryHeader = "[User Query]\n";

inline constexpr std::size_t kMaxSystemPromptChars = 1024;

} // namespace ChatPrompt

} // namespace Thoth

#endif // THOTH_CHAT_PROMPT_CONFIG_H
