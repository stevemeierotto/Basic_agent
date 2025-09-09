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

# Include directories
include_directories(include)

# Collect all source files
file(GLOB SOURCES
    src/*.cpp
    plugin.cpp
)

# Find libcurl
find_package(CURL REQUIRED)

# Create the executable
add_executable(code_agent_plugin ${SOURCES})

# Link with curl
target_link_libraries(basic_agent PRIVATE CURL::libcurl)
