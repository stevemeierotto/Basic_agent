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

# Collect all source files recursively from src/ and include main.cpp
file(GLOB_RECURSE SOURCES
    src/*.cpp
    main.cpp
    # plugin.cpp   # Uncomment if plugin.cpp should be part of basic_agent
)

# Find libcurl
find_package(CURL REQUIRED)

# Create the executable
add_executable(basic_agent ${SOURCES})

# Link with curl
target_link_libraries(basic_agent PRIVATE CURL::libcurl)

