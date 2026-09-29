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
#include <mutex>
#include <cstdint>
#include <memory>
#include <vector>
#include "config.h"
#include "generation_call.h"
#include "inference_types.h"

namespace Thoth {
class InferenceClient;
}

enum class LLMBackend {
    Ollama,
    OpenAI
};

/** @deprecated Prefer Thoth::LlmTokenUsage — kept as alias for existing call sites. */
using LlmTokenUsage = Thoth::LlmTokenUsage;

class LLMInterface {
public:
     LLMInterface(LLMBackend backend = LLMBackend::Ollama, Config* cfg = nullptr);
    ~LLMInterface();

    // Internal helpers
    std::string askOllama(const std::string& prompt);
    std::string askOllama(const std::string& prompt, int num_predict_override);
    std::string askOllama(const std::string& prompt,
                          int num_predict_override,
                          const std::vector<std::string>& stop_sequences);
    std::string askOpenAI(const std::string& prompt);

    std::string query(const std::string& prompt);
    /** @param num_predict_override Ollama/llama num_predict; -1 uses config->max_tokens. */
    std::string query(const std::string& prompt, int num_predict_override);
    /** Plan M G3 — chat path may pass stop sequences (empty = omit). */
    std::string query(const std::string& prompt,
                      int num_predict_override,
                      const std::vector<std::string>& stop_sequences);

    /**
     * Plan N N2 — structured generate for chat safety.
     * Shares mock / unavailable / test-suite gates with query().
     */
    Thoth::InferenceGenerateResult queryDetailed(
        const std::string& prompt,
        int num_predict_override,
        const std::vector<std::string>& stop_sequences);

    /** Phase A — structured chat generate via /v1/chat/completions (llama_cpp only). */
    Thoth::InferenceGenerateResult queryDetailedChat(
        const Thoth::InferenceChatRequest& request,
        int num_predict_override,
        const std::vector<std::string>& stop_sequences);

    /**
     * Call-scoped generation. The returned outcome belongs to this invocation.
     * When THOTH_GENERATION_MAX_TOKENS is set, it replaces num_predict_override.
     */
    Thoth::GenerationOutcome generateCall(
        const std::string& prompt,
        int num_predict_override,
        const std::vector<std::string>& stop_sequences,
        const Thoth::GenerationCallContext& context);

    void setInferenceClientForTests(std::unique_ptr<Thoth::InferenceClient> client);

    /** Plan N N6 — format Class A provider errors for the chat UI (no ChatGenerationResult UI strings). */
    std::string formatProviderError(const std::string& detail);

    LLMBackend getBackend() const { return backend; }
    
    // Allow switching dynamically
    void setBackend(LLMBackend backend);
    void setConfig(Config* cfg) { config = cfg; }

    void useModel(const std::string& model) { selectedModel = model; }
    const std::string& getSelectedModel() const { return selectedModel; }

    void resetSessionTokenUsage();
    LlmTokenUsage lastCallTokenUsage() const;
    LlmTokenUsage sessionTokenUsage() const;

private:
    void recordTokenUsage(const LlmTokenUsage& usage);
    static LlmTokenUsage estimateTokenUsage(const std::string& prompt, const std::string& response);
    static LlmTokenUsage parseOllamaTokenUsage(const std::string& rawJson);
    static LlmTokenUsage parseOpenAiTokenUsage(const std::string& rawJson);

    void ensureInferenceClient();
    std::string detectOllamaModel();
    std::string resolveOllamaModel();

    LLMBackend backend;
    Config* config;
    std::unique_ptr<Thoth::InferenceClient> inference_client_;
    std::string selectedModel; 
    mutable std::recursive_mutex llmMutex;
    LlmTokenUsage last_call_usage_;
    LlmTokenUsage session_usage_;
};

