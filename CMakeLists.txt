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

# List source files for the library
set(BASIC_AGENT_SOURCES
    src/command_processor.cpp
    src/config.cpp
    src/decision_trace.cpp
    src/default_planner.cpp
    src/llm_planner.cpp
    src/embedding_engine.cpp
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
    src/plan.cpp
    src/plan_parser.cpp
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
    src/benchmark_runner.cpp
    src/benchmark_reporter.cpp
    src/benchmark_case_registry.cpp
)

# Create the shared library
add_library(basic_agent ${BASIC_AGENT_SOURCES})
target_link_libraries(basic_agent PRIVATE CURL::libcurl SQLite::SQLite3)

# Benchmark Tool
add_executable(run_grag_benchmark src/run_grag_benchmark.cpp)
target_link_libraries(run_grag_benchmark PRIVATE basic_agent)

add_executable(run_cognate_benchmark src/run_cognate_benchmark.cpp)
target_link_libraries(run_cognate_benchmark PRIVATE basic_agent)

# CLI executable for debugging or standalone use
# Only build CLI tool if we're building standalone
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_executable(basic_agent_cli main.cpp)
    target_link_libraries(basic_agent_cli PRIVATE basic_agent)
endif()



