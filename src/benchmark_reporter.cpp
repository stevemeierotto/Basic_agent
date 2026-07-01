/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — BenchmarkReporter implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/benchmark_reporter.h"
#include "../include/file_handler.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <ctime>

namespace Thoth {

bool BenchmarkReporter::reportToFile(const ComparisonResult& result,
                                     int corpus_chunk_count,
                                     const BenchmarkRunIdentity& identity) {
    FileHandler fh;
    std::string path = fh.getAgentWorkspacePath("grag_benchmark.jsonl");

    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    nlohmann::json entry;
    if (!identity.empty()) {
        entry["run_id"] = identity.run_id;
        entry["env_hash"] = identity.env_hash;
    } else {
        entry["run_id"] = "benchmark-" + std::to_string(now_ms);
    }
    entry["run_at_ms"] = now_ms;
    entry["corpus_chunk_count"] = corpus_chunk_count;

    entry["rag_config"] = {{"wq", 1.0}, {"wd", 0.0}, {"wt", 0.0}, {"top_k", 5}};
    entry["grag_config"] = {{"wq", 0.4}, {"wd", 0.4}, {"wt", 0.2}, {"top_k", 5}};

    int grag_wins = 0, rag_wins = 0, ties = 0;
    for (const auto& delta : result.deltas) {
        if (delta.ndcg_delta > 0.001f) grag_wins++;
        else if (delta.ndcg_delta < -0.001f) rag_wins++;
        else ties++;
    }

    entry["summary"] = {
        {"rag_mean_precision_at_5", result.rag_mean_precision},
        {"grag_mean_precision_at_5", result.grag_mean_precision},
        {"rag_mean_reciprocal_rank", result.rag_mean_reciprocal_rank},
        {"grag_mean_reciprocal_rank", result.grag_mean_reciprocal_rank},
        {"rag_mean_ndcg_at_5", result.rag_mean_ndcg},
        {"grag_mean_ndcg_at_5", result.grag_mean_ndcg},
        {"grag_wins_ndcg", grag_wins},
        {"rag_wins_ndcg", rag_wins},
        {"ties_ndcg", ties}
    };

    nlohmann::json by_type = nlohmann::json::object();
    for (auto const& [type, prec] : result.rag_precision_by_type) {
        by_type[type]["rag_precision"] = prec;
        
        auto it_gp = result.grag_precision_by_type.find(type);
        by_type[type]["grag_precision"] = (it_gp != result.grag_precision_by_type.end()) ? it_gp->second : 0.0f;
        
        auto it_rn = result.rag_ndcg_by_type.find(type);
        by_type[type]["rag_ndcg"] = (it_rn != result.rag_ndcg_by_type.end()) ? it_rn->second : 0.0f;
        
        auto it_gn = result.grag_ndcg_by_type.find(type);
        by_type[type]["grag_ndcg"] = (it_gn != result.grag_ndcg_by_type.end()) ? it_gn->second : 0.0f;
    }
    entry["by_type"] = by_type;

    nlohmann::json cases = nlohmann::json::array();
    for (const auto& delta : result.deltas) {
        cases.push_back({
            {"case_id", delta.case_id},
            {"case_type", delta.case_type},
            {"rag_precision", delta.rag_precision},
            {"grag_precision", delta.grag_precision},
            {"precision_delta", delta.precision_delta},
            {"rag_rr", delta.rag_reciprocal_rank},
            {"grag_rr", delta.grag_reciprocal_rank},
            {"rag_ndcg", delta.rag_ndcg},
            {"grag_ndcg", delta.grag_ndcg},
            {"ndcg_delta", delta.ndcg_delta},
            {"directional_lift", delta.directional_lift}
        });
    }
    entry["cases"] = cases;

    std::ofstream out(path, std::ios::app);
    if (out.is_open()) {
        out << entry.dump() << "\n";
        
        // Phase 5.2 Auto-Archive
        archiveResults(result, corpus_chunk_count);
        
        return true;
    }
    return false;
}

void BenchmarkReporter::reportToStdout(const ComparisonResult& result) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << " GRAG vs RAG RETRIEVAL BENCHMARK SUMMARY (Metric Hardening)\n";
    std::cout << std::string(70, '=') << "\n\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "OVERALL PERFORMANCE:\n";
    std::cout << "  Mean Precision@5:  RAG: " << result.rag_mean_precision << "  GRAG: " << result.grag_mean_precision << "\n";
    std::cout << "  Mean RR:           RAG: " << result.rag_mean_reciprocal_rank << "  GRAG: " << result.grag_mean_reciprocal_rank << "\n";
    std::cout << "  Mean nDCG@5:       RAG: " << result.rag_mean_ndcg << "  GRAG: " << result.grag_mean_ndcg << "\n\n";

    std::cout << "nDCG@5 BY CASE TYPE:\n";
    std::cout << "  " << std::left << std::setw(25) << "Type" << std::setw(10) << "RAG" << std::setw(10) << "GRAG" << "Delta\n";
    std::cout << "  " << std::string(55, '-') << "\n";
    
    for (auto const& [type, rag_ndcg] : result.rag_ndcg_by_type) {
        float grag_ndcg = 0.0f;
        auto it = result.grag_ndcg_by_type.find(type);
        if (it != result.grag_ndcg_by_type.end()) grag_ndcg = it->second;
        
        float delta = grag_ndcg - rag_ndcg;
        std::cout << "  " << std::left << std::setw(25) << type << std::setw(10) << rag_ndcg << std::setw(10) << grag_ndcg 
                  << (delta > 0 ? "+" : "") << delta << "\n";
    }

    std::cout << "\nTOP 3 GRAG LIFT (Directional Rank Delta):\n";
    auto sorted_deltas = result.deltas;
    std::sort(sorted_deltas.begin(), sorted_deltas.end(), [](const auto& a, const auto& b) {
        return a.directional_lift > b.directional_lift;
    });

    for (int i = 0; i < std::min(3, (int)sorted_deltas.size()); ++i) {
        if (sorted_deltas[i].directional_lift <= 0) break;
        std::cout << "  [" << sorted_deltas[i].case_id << "] (" << sorted_deltas[i].case_type << "): Lift: +" << (int)sorted_deltas[i].directional_lift << " positions\n";
    }

    std::cout << "\nTOP 3 GRAG nDCG WINS:\n";
    std::sort(sorted_deltas.begin(), sorted_deltas.end(), [](const auto& a, const auto& b) {
        return a.ndcg_delta > b.ndcg_delta;
    });

    for (int i = 0; i < std::min(3, (int)sorted_deltas.size()); ++i) {
        if (sorted_deltas[i].ndcg_delta <= 0) break;
        std::cout << "  [" << sorted_deltas[i].case_id << "] (" << sorted_deltas[i].case_type << "): nDCG Delta: +" << sorted_deltas[i].ndcg_delta << "\n";
    }
    
    std::cout << "\n" << std::string(70, '=') << "\n\n";
}

bool BenchmarkReporter::archiveResults(const ComparisonResult& result, int corpus_chunk_count) {
    FileHandler fh;
    std::string path = fh.getProjectRoot() + "/docs/benchmark_results.md";
    
    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) return false;

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    
    const char* envModel = std::getenv("OLLAMA_EMBED_MODEL");
    std::string model = envModel ? envModel : "nomic-embed-text";

    out << "\n## Benchmark Run: " << std::ctime(&now_time);
    out << "- **Embedding Model:** " << model << "\n";
    out << "- **Corpus Size:** " << corpus_chunk_count << " chunks\n";
    out << "- **Weights:** $w_q=0.4, w_d=0.4, w_k=0.3$\n\n";

    out << "| Metric | RAG (Baseline) | GRAG (Optimized) | Delta |\n";
    out << "| :--- | :---: | :---: | :---: |\n";
    
    auto fmt_row = [&](const std::string& name, float r, float g) {
        float d = g - r;
        out << "| " << name << " | " << std::fixed << std::setprecision(3) << r << " | " << g << " | " << (d >= 0 ? "+" : "") << d << " |\n";
    };

    fmt_row("Mean Precision@5", result.rag_mean_precision, result.grag_mean_precision);
    fmt_row("Mean MRR", result.rag_mean_reciprocal_rank, result.grag_mean_reciprocal_rank);
    fmt_row("Mean nDCG@5", result.rag_mean_ndcg, result.grag_mean_ndcg);
    
    out << "\n---\n";
    return true;
}

} // namespace Thoth
