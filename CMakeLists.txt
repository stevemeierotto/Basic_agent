# Copyright (c) 2024 Steve Meierotto
#
# basic_agent - AI Agent with Memory and RAG Capabilities
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text

cmake_minimum_required(VERSION 3.10)

# Project name
project(BasicAgent LANGUAGES CXX)

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to build shared library
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Include directories
include_directories(include)

# Find libcurl and sqlite3
find_package(CURL REQUIRED)
find_package(SQLite3 REQUIRED)

if(NOT DEFINED THOTH_BUILD_ENGINE)
    set(THOTH_BUILD_ENGINE ON)
endif()

if(NOT DEFINED THOTH_ENGINE_VERSION)
    set(THOTH_ENGINE_VERSION "0.2")
endif()

# List source files for the library
set(BASIC_AGENT_SOURCES
    src/command_processor.cpp
    src/config.cpp
    src/decision_trace.cpp
    src/default_planner.cpp
    src/llm_planner.cpp
    src/embedding_engine.cpp
    src/inference_endpoint.cpp
    src/inference_client.cpp
    src/inference_http.cpp
    src/ollama_client.cpp
    src/llama_server_client.cpp
    src/mock_inference_client.cpp
    src/runtime_bootstrap.cpp
    src/env_loader.cpp
    src/flat_vector_store.cpp
    src/constraint_checker.cpp
    src/fact_store.cpp
    src/store_fact_tool.cpp
    src/self_correct_tool.cpp
    src/executive_controller.cpp
    src/file_handler.cpp
    src/grag_scorer.cpp
    src/grag_metrics.cpp
    src/index_manager.cpp
    src/llm_interface.cpp
    src/logger.cpp
    src/memory.cpp
    src/memory_pruner.cpp
    src/memory_consolidation_metrics.cpp
    src/consolidation_policy.cpp
    src/consolidation_api.cpp
    src/restore_api.cpp
    src/inference_backend_probe.cpp
    src/clock.cpp
    src/episodic_memory.cpp
    src/summary_generator.cpp
    src/plan.cpp
    src/plan_parser.cpp
    src/plan_validator.cpp
    src/chat_rag_logger.cpp
    src/project_analyze_tool.cpp
    src/run_tests_tool.cpp
    src/trajectory.cpp
    src/trajectory_builder.cpp
    src/graph_refiner.cpp
    src/strategy.cpp
    src/strategy_engine.cpp
    src/sqlite_memory_repository.cpp
    src/memory_repository_factory.cpp
    src/prompt_factory.cpp
    src/rag.cpp
    src/similarity.cpp
    src/tools.cpp
    src/workflow_engine.cpp
    src/step_metrics_repository.cpp
    src/standard_execution_mode.cpp
    src/scientific_execution_mode.cpp
    src/gmail_read_labels_tool.cpp
    src/gmail_read_messages_tool.cpp
    src/code_modify_tool.cpp
    src/vector_store.cpp
    src/web_scrape_tool.cpp
    src/chunker/chunker.cpp
    src/basic_agent_plugin.cpp
    src/engine_runtime.cpp
    src/engine_event.cpp
    src/engine_error.cpp
    src/benchmark_runner.cpp
    src/benchmark_reporter.cpp
    src/benchmark_case_registry.cpp
    src/chat_rag_golden_cases.cpp
    src/chat_retrieval_boost.cpp
    src/chat_generation_safety.cpp
    src/chat_query_utils.cpp
    src/reflection_utils.cpp
    src/reflection_ab_cases.cpp
    src/robustness_mock_responses.cpp
    src/robustness_cases.cpp
    src/test_suite_dev.cpp
    src/benchmark_environment.cpp
    src/git_metadata.cpp
    src/ollama_snapshot.cpp
    src/benchmark_context.cpp
    src/cognitive_metrics_logger.cpp
    src/episode_events.cpp
    src/episode_event_channel.cpp
    src/replay_subscriber.cpp
    src/metrics_subscriber.cpp
    src/trace_subscriber.cpp
    src/trajectory_ablation.cpp
)

if(THOTH_BUILD_ENGINE)
    list(APPEND BASIC_AGENT_SOURCES src/engine_http_transport.cpp src/engine_sse_session.cpp)
endif()

# E2 evaluation kernel — no rag.cpp / runtime heuristics (docs/E2_PROTOCOL.md § Link isolation)
set(E2_EVAL_KERNEL_SOURCES
    src/episodic_learning_eval.cpp
    src/episodic_learning_cases.cpp
    src/episodic_evaluation_service.cpp
    src/evaluation_subscriber.cpp
    src/diagnostic_service.cpp
    src/pipeline_telemetry_service.cpp
    src/e2_path_equivalence.cpp
    src/e2_strict_enforcement.cpp
    src/e2_strict_retrieval.cpp
)
add_library(e2_eval_kernel STATIC ${E2_EVAL_KERNEL_SOURCES})
target_include_directories(e2_eval_kernel PUBLIC include)
target_compile_definitions(e2_eval_kernel PUBLIC THOTH_E2_STRICT_KERNEL=1)
set_target_properties(e2_eval_kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Create the shared library
add_library(basic_agent ${BASIC_AGENT_SOURCES})
target_link_libraries(basic_agent PUBLIC e2_eval_kernel PRIVATE CURL::libcurl SQLite::SQLite3)

if(THOTH_BUILD_ENGINE)
    find_package(Threads REQUIRED)
    find_package(Git QUIET)
    set(THOTH_GIT_SHA_VALUE "unknown")
    if(GIT_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE THOTH_GIT_SHA_VALUE
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
    endif()
    if(NOT THOTH_GIT_SHA_VALUE)
        set(THOTH_GIT_SHA_VALUE "unknown")
    endif()

    target_compile_definitions(basic_agent PRIVATE
        THOTH_ENGINE_VERSION="${THOTH_ENGINE_VERSION}"
        THOTH_GIT_SHA="${THOTH_GIT_SHA_VALUE}")
    target_include_directories(basic_agent PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_link_libraries(basic_agent PRIVATE Threads::Threads)
endif()

# Benchmark Tool
add_executable(run_grag_benchmark src/run_grag_benchmark.cpp)
target_link_libraries(run_grag_benchmark PRIVATE basic_agent)

add_executable(run_cognate_benchmark src/run_cognate_benchmark.cpp)
target_link_libraries(run_cognate_benchmark PRIVATE basic_agent)

add_executable(run_chat_rag_benchmark src/run_chat_rag_benchmark.cpp)
target_link_libraries(run_chat_rag_benchmark PRIVATE basic_agent)

add_executable(run_reflection_ab_benchmark src/run_reflection_ab_benchmark.cpp)
target_link_libraries(run_reflection_ab_benchmark PRIVATE basic_agent)

add_executable(run_robustness_suite src/run_robustness_suite.cpp)
target_link_libraries(run_robustness_suite PRIVATE basic_agent)

add_executable(run_trajectory_ablation_benchmark src/run_trajectory_ablation_benchmark.cpp)
target_link_libraries(run_trajectory_ablation_benchmark PRIVATE basic_agent)

add_executable(run_episodic_learning_benchmark src/run_episodic_learning_benchmark.cpp)
target_compile_definitions(run_episodic_learning_benchmark PRIVATE THOTH_E2_OFFICIAL_HARNESS=1)
target_link_libraries(run_episodic_learning_benchmark PRIVATE basic_agent e2_eval_kernel)

if(THOTH_BUILD_ENGINE)
    add_executable(thoth-engine src/thoth_engine_main.cpp)
    target_link_libraries(thoth-engine PRIVATE basic_agent)
    target_compile_definitions(thoth-engine PRIVATE
        THOTH_ENGINE_VERSION="${THOTH_ENGINE_VERSION}")
    set_target_properties(thoth-engine PROPERTIES FOLDER "apps")
endif()

# CLI executable for debugging or standalone use
# Only build CLI tool if we're building standalone
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_executable(basic_agent_cli main.cpp)
    target_link_libraries(basic_agent_cli PRIVATE basic_agent)
endif()



