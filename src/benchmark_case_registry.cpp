/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — BenchmarkCaseRegistry implementation
 * Generated with 100 cases.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/benchmark_case_registry.h"

namespace Thoth {

std::vector<BenchmarkCase> Thoth::BenchmarkCaseRegistry::getCases() {
    std::vector<BenchmarkCase> cases;
    cases.push_back({"U1", "N/A", "alpha blending formula and direction magnitude magnitude = ||G - C||", "", {"GRAG.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U2", "N/A", "rescore formula score = HybridVectorScore + (wk * KeywordScore) + (wg * GraphScore)", "", {"GRAG.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U3", "N/A", "GRAG weights wq=0.4, wd=0.4, wt=0.2, wk=0.3, wg=0.3", "", {"GRAG.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U4", "N/A", "Multi-Index Routing Modes PLAN_AWARE GOAL_ONLY CONVERSATIONAL", "", {"GRAG.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U5", "N/A", "ControllerState enum IDLE PLANNING EXECUTING_STEP OBSERVING_RESULT", "", {"PLAN.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U6", "N/A", "IExecutionMode interface StandardExecutionMode ScientificExecutionMode", "", {"PLAN.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U7", "N/A", "IPlanner interface create_plan revise_plan", "", {"PLAN.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U8", "N/A", "ExecutiveController core identity resumable observable goal-driven state machine", "", {"PLAN.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U9", "N/A", "Cognate framework high-level cognitive framework perception memory action cycle", "", {"cognate.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U10", "N/A", "CognateNode class manages its own local memory and state", "", {"cognate.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U11", "N/A", "the Cognate loop represents the iterative process of thought and action", "", {"cognate.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U12", "N/A", "Step 3.1 project_analyze tool scans Thoth source directory", "", {"improvements.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U13", "N/A", "Step 3.2 run_tests tool executes unit test suite", "", {"improvements.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U14", "N/A", "Step 3.3 code_modify tool apply_diff unified_diff", "", {"improvements.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U15", "N/A", "Step 4.2 MemoryPruner class implementation transactional SQLite", "", {"improvements.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U16", "N/A", "Step 4.3 Structured Fact Store facts SQLite table", "", {"improvements.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U17", "N/A", "NODE system execution wiring layer n8n-style workflow engine", "", {"NODE.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U18", "N/A", "NODE types InputNode RetrievalNode PromptAssemblyNode LLMNode ToolNode OutputNode", "", {"NODE.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U19", "N/A", "NODE principles black-box execution units deterministic inputs", "", {"NODE.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U20", "N/A", "Event Emission Integrity Audit emit_event calls", "", {"architectural_facts.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U21", "N/A", "thread safety in Memory class shared_mutex data race prevention", "", {"architectural_facts.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U22", "N/A", "Trajectory resume compatibility event metadata recording", "", {"architectural_facts.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U23", "N/A", "2026-03-10 Strict Sandbox Boundaries IndexManager hard-reject", "", {"completed_improvements_log.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U24", "N/A", "2026-03-10 Trajectory-Aware Retrieval Phase 5.5 TrajectoryBuilder", "", {"completed_improvements_log.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U25", "N/A", "2026-03-09 Reranking Optimization Candidate Expansion 40 candidates", "", {"completed_improvements_log.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U26", "N/A", "AI Coding Agent Guide architecture conventions critical rules", "", {"AGENTS.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U27", "N/A", "rules for AI agents DO NOT modify agent_workspace files directly", "", {"AGENTS.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U28", "N/A", "naming conventions snake_case filenames PascalCase class names", "", {"AGENTS.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U29", "N/A", "AgentInterface bridge between GUI and core library", "", {"AGENTS.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U30", "N/A", "THRESHOLD value 0.3 for alpha clamping", "", {"GRAG.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U31", "N/A", "StepFailurePolicy max_retries abort_on_failure", "", {"PLAN.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U32", "N/A", "Step 4.4 Vector Store Scalability IVectorStore interface", "", {"improvements.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U33", "N/A", "Topological order node dependency resolution", "", {"NODE.md"}, "UNAMBIGUOUS"});
    cases.push_back({"U34", "N/A", "GUI layer src directory wxWidgets-based MainFrame", "", {"AGENTS.md"}, "UNAMBIGUOUS"});
    cases.push_back({"G1", "Understand future development plans", "details on planning", "", {"improvements.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G2", "Understand the code structure of plans", "details on planning", "", {"PLAN.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G3", "Understand the cognitive loop of planning", "details on planning", "", {"cognate.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G4", "Optimize retrieval scoring signals", "memory system", "", {"GRAG.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G5", "Review thread safety and mutexes", "memory system", "", {"architectural_facts.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G6", "Learn about pruning and archival", "memory system", "", {"improvements.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G7", "Learn about the perception-action cycle", "execution logic", "", {"cognate.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G8", "Understand how steps are run by the controller", "execution logic", "", {"PLAN.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G9", "Review the workflow engine wiring", "execution logic", "", {"NODE.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G10", "Configure the hybrid scoring formula", "retrieval details", "", {"GRAG.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G11", "Build the vector store abstraction", "retrieval details", "", {"improvements.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G12", "Follow the RAG pipeline conventions", "retrieval details", "", {"AGENTS.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G13", "Set the weight for past actions", "trajectory usage", "", {"GRAG.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G14", "Ensure resume compatibility after crash", "trajectory usage", "", {"architectural_facts.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G15", "Verify implementation of progress awareness", "trajectory usage", "", {"completed_improvements_log.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G16", "Compose a new visual workflow", "node system", "", {"NODE.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G17", "Define a new cognitive agent framework", "node system", "", {"cognate.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G18", "Dispatch a task to the execution harness", "node system", "", {"PLAN.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G19", "Follow the strict coding standards", "system rules", "", {"AGENTS.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G20", "Understand the deterministic execution principles", "system rules", "", {"NODE.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G21", "Implement the sandbox path enforcement", "system rules", "", {"improvements.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G22", "Check which features were finished recently", "latest updates", "", {"completed_improvements_log.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G23", "Check what work is planned next", "latest updates", "", {"improvements.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G24", "Review the current scoring version", "latest updates", "", {"GRAG.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G25", "How does the agent decide what to do next", "decision process", "", {"PLAN.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G26", "How does the agent process perception into action", "decision process", "", {"cognate.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G27", "How are node dependencies resolved", "decision process", "", {"NODE.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G28", "Is the memory class thread safe", "safety and integrity", "", {"architectural_facts.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G29", "What are the forbidden actions for agents", "safety and integrity", "", {"AGENTS.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G30", "How is the sandbox boundary enforced", "safety and integrity", "", {"improvements.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G31", "What is the formula for direction", "vector math", "", {"GRAG.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G32", "What is the dimension of the embeddings", "vector math", "", {"improvements.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"G33", "How are embeddings stored in SQLite", "vector math", "", {"architectural_facts.md"}, "GOAL_DISAMBIGUATES"});
    cases.push_back({"T1", "Maintain system integrity", "rules for changes", "I'm about to modify the core library.", {"AGENTS.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T2", "Maintain system integrity", "rules for changes", "I'm looking at the roadmap for Phase 4.", {"improvements.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T3", "Improve retrieval quality", "scoring weights", "I just analyzed the query similarity Q.", {"GRAG.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T4", "Improve retrieval quality", "scoring weights", "I'm checking the future plan for weight tuning.", {"improvements.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T5", "Analyze task execution", "state machine", "I'm looking at how plans are created.", {"PLAN.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T6", "Analyze task execution", "state machine", "I'm debugging event emission in the controller.", {"architectural_facts.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T7", "Understand cognitive flow", "planning loop", "I'm studying the high-level Cognate architecture.", {"cognate.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T8", "Understand cognitive flow", "planning loop", "I'm reading about the IPlanner interface.", {"PLAN.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T9", "Understand cognitive flow", "planning loop", "I'm reviewing the Roadmap for Phase 5.", {"improvements.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T10", "Review memory implementation", "SQLite details", "I just finished checking 2026-03-10 updates.", {"completed_improvements_log.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T11", "Review memory implementation", "SQLite details", "I'm looking at the Fact Store design in Phase 4.", {"improvements.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T12", "Review memory implementation", "SQLite details", "I'm auditing thread safety in the Memory class.", {"architectural_facts.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T13", "Configure workflow engine", "node types", "I'm designing a new n8n-style graph.", {"NODE.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T14", "Configure workflow engine", "node types", "I'm adding a NODE type to the PlanStep struct.", {"PLAN.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T15", "Analyze directional retrieval", "alpha value", "I'm calculating the magnitude of D.", {"GRAG.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T16", "Analyze directional retrieval", "alpha value", "I'm checking if adaptive tuning is on the roadmap.", {"improvements.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T17", "Follow project conventions", "file structure", "I'm organizing my new source files.", {"AGENTS.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T18", "Follow project conventions", "file structure", "I'm seeing what files were added on March 5th.", {"completed_improvements_log.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T19", "Manage agent lifecycle", "transition logic", "I'm defining the ControllerState enum.", {"PLAN.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T20", "Manage agent lifecycle", "transition logic", "I'm verifying event sequence timing.", {"architectural_facts.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T21", "Scale agent intelligence", "framework upgrades", "I'm looking at the CognateNode expansion.", {"cognate.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T22", "Scale agent intelligence", "framework upgrades", "I'm reading the Phase 5 advanced reasoning roadmap.", {"improvements.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T23", "Verify retrieval math", "formula components", "I'm looking at the GraphScore term.", {"GRAG.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T24", "Verify retrieval math", "formula components", "I'm inspecting the output of a RetrievalNode.", {"NODE.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T25", "Handle plan failures", "retry logic", "I'm configuring max_retries per step.", {"PLAN.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T26", "Handle plan failures", "retry logic", "I'm checking the self-building capability in Phase 3.", {"improvements.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T27", "Audit system behavior", "log analysis", "I'm tracing emit_event calls.", {"architectural_facts.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T28", "Audit system behavior", "log analysis", "I'm reviewing the 2026-03-09 performance run.", {"completed_improvements_log.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T29", "Review coding rules", "forbidden actions", "I'm about to touch the GUI files.", {"AGENTS.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T30", "Review coding rules", "forbidden actions", "I'm checking sandbox enforcement in Phase 3.", {"improvements.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T31", "Design modular units", "interface rules", "I'm defining a new Node input schema.", {"NODE.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T32", "Design modular units", "interface rules", "I'm implementing the IExecutionMode strategy.", {"PLAN.md"}, "TRAJECTORY_DISAMBIGUATES"});
    cases.push_back({"T33", "Design modular units", "interface rules", "I'm reviewing the perception-action loop.", {"cognate.md"}, "TRAJECTORY_DISAMBIGUATES"});

    return cases;
}

} // namespace Thoth
