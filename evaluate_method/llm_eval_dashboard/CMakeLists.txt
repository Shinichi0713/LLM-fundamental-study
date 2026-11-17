cmake_minimum_required(VERSION 3.20)

project(llm_eval_dashboard VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LLMBENCH_BUILD_TESTS "Build unit tests" ON)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(llmbench_core STATIC
  src/benchmark.cpp
  src/demo.cpp
  src/http_client.cpp
  src/mock_provider.cpp
  src/openai_client.cpp
  src/runner.cpp
  src/storage.cpp
  src/types.cpp
  src/util.cpp
)

target_include_directories(llmbench_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_compile_definitions(llmbench_core PUBLIC
  NOMINMAX
  WIN32_LEAN_AND_MEAN
  _CRT_SECURE_NO_WARNINGS
)

if(WIN32)
  target_link_libraries(llmbench_core PUBLIC winhttp)
else()
  find_package(Threads REQUIRED)
  target_link_libraries(llmbench_core PUBLIC Threads::Threads)

  # HTTPS support for the reference HTTP client is optional on POSIX.
  find_package(OpenSSL QUIET)
  if(OpenSSL_FOUND)
    target_compile_definitions(llmbench_core PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
    target_link_libraries(llmbench_core PUBLIC OpenSSL::SSL OpenSSL::Crypto)
  endif()
endif()

if(MSVC)
  target_compile_options(llmbench_core PRIVATE /W3 /utf-8 /EHsc)
else()
  target_compile_options(llmbench_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# Embed the dashboard assets so the executable also works standalone
# ---------------------------------------------------------------------------
set(EMBEDDED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/embedded_assets_data.h)
file(GLOB WEB_ASSETS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/web/*)
add_custom_command(
  OUTPUT ${EMBEDDED_HEADER}
  COMMAND ${CMAKE_COMMAND}
          -DWEB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/web
          -DOUTPUT=${EMBEDDED_HEADER}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_assets.cmake
  DEPENDS ${WEB_ASSETS} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_assets.cmake
  COMMENT "Embedding dashboard assets"
  VERBATIM
)

add_library(llmbench_assets STATIC src/embedded_assets.cpp ${EMBEDDED_HEADER})
target_include_directories(llmbench_assets PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(llmbench_assets PUBLIC llmbench_core)
if(MSVC)
  # The generated header embeds UTF-8 dashboard assets.
  target_compile_options(llmbench_assets PRIVATE /utf-8)
endif()

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(llmbench src/main.cpp src/app.cpp)
target_link_libraries(llmbench PRIVATE llmbench_assets llmbench_core)
target_include_directories(llmbench PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)

if(MSVC)
  target_compile_options(llmbench PRIVATE /W3 /utf-8 /EHsc)
else()
  target_compile_options(llmbench PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Copy data next to the executable for out-of-the-box use.
add_custom_command(TARGET llmbench POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/web $<TARGET_FILE_DIR:llmbench>/web
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/data $<TARGET_FILE_DIR:llmbench>/data
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/configs $<TARGET_FILE_DIR:llmbench>/configs
  COMMENT "Copying dashboard assets"
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(LLMBENCH_BUILD_TESTS)
  enable_testing()
  add_executable(llmbench_tests tests/test_core.cpp)
  target_link_libraries(llmbench_tests PRIVATE llmbench_core)
  target_include_directories(llmbench_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  if(MSVC)
    target_compile_options(llmbench_tests PRIVATE /utf-8 /EHsc)
  endif()
  add_test(NAME core_tests COMMAND llmbench_tests)
endif()
