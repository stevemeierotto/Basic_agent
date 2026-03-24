/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — BenchmarkCaseRegistry implementation
 * Statistically Hardened Suite (100 cases)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/benchmark_case_registry.h"

namespace Thoth {

std::vector<BenchmarkCase> Thoth::BenchmarkCaseRegistry::getCases() {
    std::vector<BenchmarkCase> cases;
    cases.push_back({"U1", "N/A", "Maximum Inner Product Search MIPS for dense vector retrieval", "", {"2005.11401v4.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U2", "N/A", "virtual context management hierarchical memory tiers OS", "", {"2310.08560v2.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U3", "N/A", "Synergizing reasoning and acting in language models", "", {"2210.03629v3.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U4", "N/A", "interactive simulacra of human behavior The Sims sandbox", "", {"2304.03442v2.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U5", "N/A", "PaLM 540B striking gains on GSM8K benchmark", "", {"2201.11903v6.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U6", "N/A", "marginalize over latent documents top-K approximation", "", {"2005.11401v4.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U7", "N/A", "FIFO queue rolling history context eviction", "", {"2310.08560v2.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U8", "N/A", "Search Lookup and Finish environment interface", "", {"2210.03629v3.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U9", "N/A", "recency importance and relevance memory retrieval", "", {"2304.03442v2.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U10", "N/A", "series of intermediate reasoning steps emerge naturally", "", {"2201.11903v6.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U11", "N/A", "DPR dense passage retriever Wikipedia snippets", "", {"2005.11401v4.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U12", "N/A", "paging and disk swaps for context management", "", {"2310.08560v2.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U13", "N/A", "HotpotQA and StrategyQA experimental results", "", {"2210.03629v3.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U14", "N/A", "Hobbs Cafe and generative agent architecture", "", {"2304.03442v2.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U15", "N/A", "few-shot prompting reasoning chain exemplars", "", {"2201.11903v6.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U16", "N/A", "parametric and non-parametric memory components", "", {"2005.11401v4.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U17", "N/A", "fixed-length context window as physical memory", "", {"2310.08560v2.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U18", "N/A", "interleaving thoughts and actions for decision making", "", {"2210.03629v3.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U19", "N/A", "reflection engine for agent generalization", "", {"2304.03442v2.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"U20", "N/A", "large scale language model reasoning capabilities", "", {"2201.11903v6.txt"}, "UNAMBIGUOUS"});
    cases.push_back({"G1", "Manage limited context windows in long conversations", "hierarchical memory system", "", {"2310.08560v2.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G2", "Model emergent social behavior in artificial societies", "hierarchical memory system", "", {"2304.03442v2.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G3", "Interact with external environments via search", "reasoning and acting", "", {"2210.03629v3.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G4", "Simulate daily behaviors like cooking and working", "reasoning and acting", "", {"2304.03442v2.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G5", "Augment LLM knowledge with external document indices", "retrieval process", "", {"2005.11401v4.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G6", "Surface relevant past experiences for social agents", "retrieval process", "", {"2304.03442v2.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G7", "Improve arithmetic and symbolic reasoning performance", "chain of thought", "", {"2201.11903v6.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G8", "Induce and update dynamic action plans", "chain of thought", "", {"2210.03629v3.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G9", "Maintain long-term coherence in agent personas", "reflection and planning", "", {"2304.03442v2.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G10", "Explain the series of steps taken to solve a problem", "reflection and planning", "", {"2201.11903v6.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G11", "Optimize memory usage for OS-style agents", "context management", "", {"2310.08560v2.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G12", "Improve decision making in dynamic environments", "context management", "", {"2210.03629v3.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G13", "Track the behavior of thousands of simulated humans", "agent observation", "", {"2304.03442v2.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G14", "Debug the reasoning trace of an agent", "agent observation", "", {"2210.03629v3.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G15", "Build a high-performance semantic search engine", "dense vector indices", "", {"2005.11401v4.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G16", "Store massive conversation history on disk", "dense vector indices", "", {"2310.08560v2.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G17", "Solve complex math word problems", "step-by-step logic", "", {"2201.11903v6.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G18", "Navigate a house to find a specific object", "step-by-step logic", "", {"2210.03629v3.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G19", "Create believable NPC routines in a game", "daily schedules", "", {"2304.03442v2.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G20", "Execute a multi-stage search query", "daily schedules", "", {"2210.03629v3.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G21", "Implement a RAG-based question answering system", "retrieval mechanism", "", {"2005.11401v4.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G22", "Implement an infinite-context agent", "retrieval mechanism", "", {"2310.08560v2.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G23", "Study how rumors spread in a virtual town", "social dynamics", "", {"2304.03442v2.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G24", "Collaborate between multiple reasoning threads", "social dynamics", "", {"2210.03629v3.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G25", "Study the emergence of symbolic reasoning", "model scaling", "", {"2201.11903v6.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G26", "Benchmark large language models on NLP tasks", "model scaling", "", {"2005.11401v4.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G27", "Manage the internal state of a complex AI", "operating system", "", {"2310.08560v2.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G28", "Use an LLM as a controller for a robot", "operating system", "", {"2210.03629v3.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G29", "Generate high-fidelity human simulations", "behavioral modeling", "", {"2304.03442v2.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G30", "Analyze the failure modes of large models", "behavioral modeling", "", {"2201.11903v6.txt"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"T1", "Implement autonomous agents", "planning architecture", "I just finished reviewing the memory stream mechanism.", {"2304.03442v2.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T2", "Implement autonomous agents", "planning architecture", "I am currently defining the environment Search API.", {"2210.03629v3.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T3", "Integrate external information", "retrieval mechanism", "I just analyzed the MIPS dense vector search.", {"2005.11401v4.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T4", "Integrate external information", "retrieval mechanism", "I am looking at OS-inspired paging and disk swaps.", {"2310.08560v2.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T5", "Analyze agent reasoning", "intermediate thoughts", "We are testing zero-shot arithmetic word problems.", {"2201.11903v6.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T6", "Analyze agent reasoning", "intermediate thoughts", "I am tracking environmental observations in the trace.", {"2210.03629v3.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T7", "Optimize agent performance", "memory tiers", "We are focusing on social coordination in a sandbox.", {"2304.03442v2.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T8", "Optimize agent performance", "memory tiers", "I'm auditing context eviction and FIFO queues.", {"2310.08560v2.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T9", "Enable complex behaviors", "acting and behavior", "I just read about the 'marginalize over latent docs' approach.", {"2005.11401v4.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T10", "Enable complex behaviors", "acting and behavior", "The agent just decided to head to Hobbs Cafe for lunch.", {"2304.03442v2.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T11", "Scale context handling", "memory management", "I have implemented the virtual context manager.", {"2310.08560v2.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T12", "Scale context handling", "memory management", "I am currently building the reflection synthesis engine.", {"2304.03442v2.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T13", "Solve reasoning tasks", "action sequence", "I just emitted a 'Thought' about the search results.", {"2210.03629v3.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T14", "Solve reasoning tasks", "action sequence", "I am decomposing the 'wake up' plan into sub-tasks.", {"2304.03442v2.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T15", "Improve retrieval accuracy", "document ranking", "I am optimizing the DPR (Dense Passage Retriever) component.", {"2005.11401v4.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T16", "Improve retrieval accuracy", "document ranking", "I am calculating the importance score for a memory.", {"2304.03442v2.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T17", "Verify reasoning quality", "rationales", "I am providing 8-shot examples to the 540B model.", {"2201.11903v6.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T18", "Verify reasoning quality", "rationales", "I am evaluating the ALFWorld trajectory success.", {"2210.03629v3.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T19", "Manage agent memory", "tiered storage", "I just moved a block from main to external storage.", {"2310.08560v2.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T20", "Manage agent memory", "tiered storage", "I'm looking at the non-parametric memory implementation.", {"2005.11401v4.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T21", "Improve tool use", "external search", "I'm debugging the 'Lookup' command in the trace.", {"2210.03629v3.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T22", "Improve tool use", "external search", "I'm reviewing how the generator uses the retrieved context.", {"2005.11401v4.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T23", "Model personality", "character traits", "I just initialized John Lin as a pharmacy shopkeeper.", {"2304.03442v2.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T24", "Model personality", "character traits", "I'm checking the core memory section for persona persistence.", {"2310.08560v2.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T25", "Study model reasoning", "chain of thought", "I am looking at the LaMDA math results.", {"2201.11903v6.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T26", "Study model reasoning", "chain of thought", "I am interleaved reasoning and acting steps in the prompt.", {"2210.03629v3.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T27", "Optimize NLP tasks", "dense retrieval", "I'm training the bi-encoder for MIPS compatibility.", {"2005.11401v4.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T28", "Optimize NLP tasks", "dense retrieval", "I'm using the function-calling API to fetch more history.", {"2310.08560v2.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T29", "Coordinate social agents", "interaction logs", "The agent is discussing Shakespeare with Ayesha.", {"2304.03442v2.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T30", "Coordinate social agents", "interaction logs", "I'm parsing the environment response for multi-step goals.", {"2210.03629v3.txt"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"D1", "System-level tool reasoning", "memory architecture", "Query shares terms with GEN_AGENTS but Goal is OS-style", {"2310.08560v2.txt"}, "DISTRACTOR_NOISE"});
    cases.push_back({"D2", "Believable social simulations", "memory architecture", "Query shares terms with MEMGPT but Goal is character stream", {"2304.03442v2.txt"}, "DISTRACTOR_NOISE"});
    cases.push_back({"D3", "Search-driven task automation", "planning structure", "Query shares terms with GEN_AGENTS but Goal is environment acting", {"2210.03629v3.txt"}, "DISTRACTOR_NOISE"});
    cases.push_back({"D4", "Daily schedule generation", "planning structure", "Query shares terms with REACT but Goal is hourly routine", {"2304.03442v2.txt"}, "DISTRACTOR_NOISE"});
    cases.push_back({"D5", "Latent document retrieval", "knowledge retrieval", "Query shares terms with MEMGPT but Goal is dense indexing", {"2005.11401v4.txt"}, "DISTRACTOR_NOISE"});
    cases.push_back({"D6", "Paging conversation history", "knowledge retrieval", "Query shares terms with RAG but Goal is OS virtual memory", {"2310.08560v2.txt"}, "DISTRACTOR_NOISE"});
    cases.push_back({"D7", "Mathematical step reasoning", "logical chain", "Query shares terms with REACT but Goal is few-shot prompting", {"2201.11903v6.txt"}, "DISTRACTOR_NOISE"});
    cases.push_back({"D8", "Trace-based plan updates", "logical chain", "Query shares terms with COT but Goal is environmental acting", {"2210.03629v3.txt"}, "DISTRACTOR_NOISE"});
    cases.push_back({"D9", "Synthesized inferences", "reflection process", "Query shares terms with REACT but Goal is social generalization", {"2304.03442v2.txt"}, "DISTRACTOR_NOISE"});
    cases.push_back({"D10", "Internal inner speech", "reflection process", "Query shares terms with GEN_AGENTS but Goal is self-regulation", {"2210.03629v3.txt"}, "DISTRACTOR_NOISE"});
    cases.push_back({"M1", "Simulate interaction", "social coherence", "I just reviewed the reflection engine. Need to know how it affects interactions.", {"2304.03442v2.txt"}, "MULTI_HOP"});
    cases.push_back({"M2", "Complex QA", "reasoning synergy", "Search API defined. How do thoughts and actions combine for HotpotQA?", {"2210.03629v3.txt"}, "MULTI_HOP"});
    cases.push_back({"M3", "Infinite conversation", "memory hierarchy", "Context manager implemented. How do tiers interact with context window?", {"2310.08560v2.txt"}, "MULTI_HOP"});
    cases.push_back({"M4", "Knowledge intensive tasks", "conditioning generation", "Retriever trained. How does the generator use multiple retrieved docs?", {"2005.11401v4.txt"}, "MULTI_HOP"});
    cases.push_back({"M5", "Mathematical reasoning", "reasoning elicitation", "Few-shot examples provided. Why do large models gain more from chains?", {"2201.11903v6.txt"}, "MULTI_HOP"});
    cases.push_back({"M6", "ALFWorld task", "trajectory success", "Thought emitted. Action taken. How does observation affect the next thought?", {"2210.03629v3.txt"}, "MULTI_HOP"});
    cases.push_back({"M7", "Agent routinely", "recursive decomposition", "Daily plan created. How does it reach moment-to-moment behaviors?", {"2304.03442v2.txt"}, "MULTI_HOP"});
    cases.push_back({"M8", "Persistent persona", "core memory", "Session started. How does the agent update its own persistent facts?", {"2310.08560v2.txt"}, "MULTI_HOP"});
    cases.push_back({"M9", "Dense retrieval", "latent variables", "Bi-encoder trained. How is the final sequence probability calculated?", {"2005.11401v4.txt"}, "MULTI_HOP"});
    cases.push_back({"M10", "Symbolic reasoning", "scaling effects", "Tested on 8B model. Why do gains only appear above a certain scale?", {"2201.11903v6.txt"}, "MULTI_HOP"});

    return cases;
}

} // namespace Thoth
