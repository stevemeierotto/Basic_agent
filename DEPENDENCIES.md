# Dependencies

This document lists all third-party libraries and services used by basic_agent.

## Third-Party Libraries

### nlohmann/json (json.hpp)
- **Version**: Latest (header-only)
- **License**: MIT License
- **Purpose**: JSON parsing and serialization for API communication and data storage
- **Source**: https://github.com/nlohmann/json
- **License File**: `third_party/json/LICENSE.MIT`

### libcurl
- **Version**: 7.68.0+ (system library)
- **License**: MIT-style (curl license)
- **Purpose**: HTTP/HTTPS requests to OpenAI API and Ollama
- **Source**: https://curl.se/libcurl/
- **License File**: `third_party/curl/COPYING`

## External Services

### OpenAI API
- **Service**: OpenAI GPT API
- **Purpose**: Language model inference for chat responses
- **License**: Commercial API service (no code distribution)
- **Documentation**: https://platform.openai.com/docs
- **Note**: Requires API key authentication

### Ollama
- **Version**: Latest
- **License**: Apache License 2.0
- **Purpose**: Local language model inference
- **Source**: https://github.com/ollama/ollama
- **License File**: `third_party/ollama/LICENSE`
- **Note**: Used as external service, not embedded code

## Build Dependencies

### CMake
- **Version**: 3.20+
- **License**: BSD-3-Clause
- **Purpose**: Build system
- **Source**: https://cmake.org/

### C++20 Compiler
- **Supported**: GCC 10+, Clang 12+, MSVC 2019+
- **Purpose**: Compilation target
- **Standard**: ISO/IEC 14882:2020

## Installation Requirements

### System Libraries
The following system libraries must be installed:

**Ubuntu/Debian:**
```bash
sudo apt-get install libcurl4-openssl-dev cmake build-essential
```

**macOS (Homebrew):**
```bash
brew install curl cmake
```

**Windows (vcpkg):**
```bash
vcpkg install curl
```

### Header-Only Libraries
These are included directly in the project:
- `include/json.hpp` - nlohmann/json single header

## License Compatibility

All dependencies are compatible with the MIT License used by basic_agent:

- **MIT License**: json.hpp, libcurl - ✅ Compatible
- **Apache 2.0**: Ollama - ✅ Compatible (more restrictive but compatible)
- **Commercial Service**: OpenAI API - ✅ No license conflicts

## Attribution Requirements

### nlohmann/json
```
Copyright (c) 2013-2022 Niels Lohmann
Licensed under the MIT License
```

### libcurl
```
Copyright (c) 1996 - 2023, Daniel Stenberg and contributors
Licensed under the curl license (MIT-style)
```

### Ollama
```
Copyright (c) Ollama Inc.
Licensed under the Apache License, Version 2.0
```

## Updating Dependencies

To update dependencies:

1. **json.hpp**: Download latest single header from https://github.com/nlohmann/json/releases
2. **libcurl**: Update through system package manager
3. **Ollama**: Update through official installation method
4. **Update this file** with new version numbers and license information

## Security Considerations

- **OpenAI API**: Credentials stored in environment variables, never in code
- **libcurl**: Always use HTTPS endpoints, verify SSL certificates
- **Ollama**: Local service, ensure proper network security if exposing externally

## Compliance Notes

- All license files are preserved in the `third_party/` directory
- This project complies with all dependency license requirements
- No GPL or other copyleft licensed code is used
- Commercial use is permitted for all dependencies

---

*Last updated: September 2025*

