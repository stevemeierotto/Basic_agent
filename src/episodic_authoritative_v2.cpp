/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C6.4 successor episodic evaluation (episodic_authoritative_v2)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "../include/episodic_authoritative_v2.h"

#include "../include/episodic_learning_cases.h"
#include "../include/inference_endpoint.h"

#include <cctype>

namespace Thoth {

namespace {

bool isHex64(const std::string& value) {
    if (value.size() != 64) {
        return false;
    }
    for (unsigned char c : value) {
        if (!std::isxdigit(c)) {
            return false;
        }
    }
    return true;
}

bool nonEmpty(const std::string& value) {
    return !value.empty();
}

EpisodicV2Report invalid(const std::string& reason, const InferenceClient* client) {
    EpisodicV2Report report;
    report.authoritative_valid = false;
    report.status = "invalid";
    report.reason = reason;
    report.evidence = {
        {"evaluation_id", kEpisodicAuthoritativeV2Id},
        {"status", "invalid"},
        {"reason", reason},
        {"provider_observed", client ? client->backendName() : ""},
    };
    return report;
}

} // namespace

std::unique_ptr<InferenceClient> makeEpisodicV2InferenceClient(const Config* config) {
    if (config) {
        return createInferenceClient(resolveInferenceEndpoints(*config), config);
    }
    return createInferenceClient(resolveInferenceEndpoints(), nullptr);
}

EpisodicV2Report buildEpisodicAuthoritativeV2Report(const EpisodicV2Request& request,
                                                    const InferenceClient* client) {
    if (!client) {
        return invalid("inference client required", nullptr);
    }
    const std::string observed = client->backendName();
    if (!nonEmpty(observed) || observed != request.inference_backend_name) {
        return invalid("provider mismatch", client);
    }
    if (request.evaluation_tier == "mock") {
        EpisodicV2Report report;
        report.authoritative_valid = false;
        report.status = "mock";
        report.reason = "mock execution is not authoritative evidence";
        report.evidence = {
            {"evaluation_id", kEpisodicAuthoritativeV2Id},
            {"evaluation_tier", "mock"},
            {"status", "mock"},
            {"inference", {{"backend_name", observed}}},
        };
        return report;
    }
    if (request.evaluation_tier != "authoritative") {
        return invalid("evaluation_tier is not authoritative", client);
    }
    if (request.environment_schema_version != "c64-env-1" ||
        request.protocol_version != "C6.4 v1.0" || request.metric_schema_version != "1.0") {
        return invalid("C6.4 identity version mismatch", client);
    }
    if (!nonEmpty(request.llm_model) || !nonEmpty(request.embedding_model) ||
        !nonEmpty(request.embedding_method) || request.embedding_dimension <= 0 ||
        !nonEmpty(request.thoth_git_sha) || !nonEmpty(request.basic_agent_git_sha) ||
        !nonEmpty(request.corpus_fingerprint) || !isHex64(request.c64_cohort_fingerprint)) {
        return invalid("missing required identity", client);
    }

    const auto workload = getEpisodicLearningCases();
    if (request.cases.size() != workload.size()) {
        return invalid("incomplete paired comparison", client);
    }
    for (std::size_t i = 0; i < workload.size(); ++i) {
        const auto& got = request.cases[i];
        if (got.case_id != workload[i].id || !got.baseline.present || !got.episodic.present ||
            !got.baseline.response_valid || !got.episodic.response_valid) {
            return invalid("incomplete paired comparison", client);
        }
    }

    nlohmann::json cases = nlohmann::json::array();
    for (const auto& item : request.cases) {
        cases.push_back({
            {"case_id", item.case_id},
            {"workload", kEpisodicAuthoritativeV2Id},
            {"baseline", {{"condition", "baseline"}, {"response_valid", item.baseline.response_valid}}},
            {"episodic", {{"condition", "episodic"}, {"response_valid", item.episodic.response_valid}}},
        });
    }

    EpisodicV2Report report;
    report.authoritative_valid = true;
    report.status = "authoritative_recorded";
    report.reason.clear();
    report.evidence = {
        {"evaluation_id", kEpisodicAuthoritativeV2Id},
        {"evaluation_tier", "authoritative"},
        {"status", "authoritative_recorded"},
        {"environment_schema_version", request.environment_schema_version},
        {"protocol_version", request.protocol_version},
        {"metric_schema_version", request.metric_schema_version},
        {"inference", {{"backend_name", observed}}},
        {"model",
         {{"llm_model", request.llm_model},
          {"embedding_model", request.embedding_model},
          {"embedding_method", request.embedding_method},
          {"embedding_dimension", request.embedding_dimension}}},
        {"prov",
         {{"thoth_git_sha", request.thoth_git_sha},
          {"basic_agent_git_sha", request.basic_agent_git_sha}}},
        {"corpus", {{"fingerprint", request.corpus_fingerprint}}},
        {"c64_cohort_fingerprint", request.c64_cohort_fingerprint},
        {"cases", cases},
    };
    return report;
}

} // namespace Thoth
