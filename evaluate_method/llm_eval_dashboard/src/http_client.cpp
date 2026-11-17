#include "http_client.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <sstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winhttp.h>
#else
#include "httplib.h"
#endif

namespace llmbench {

namespace {

std::string lower_ascii(const std::string& value) {
  std::string result = value;
  std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return result;
}

#ifdef _WIN32

std::wstring to_wide(const std::string& value) {
  if (value.empty()) {
    return {};
  }
  int length = MultiByteToWideChar(CP_UTF8, 0, value.data(),
                                   static_cast<int>(value.size()), nullptr, 0);
  if (length <= 0) {
    return {};
  }
  std::wstring result(static_cast<std::size_t>(length), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
                      result.data(), length);
  return result;
}

std::string to_utf8(const std::wstring& value) {
  if (value.empty()) {
    return {};
  }
  int length = WideCharToMultiByte(CP_UTF8, 0, value.data(),
                                   static_cast<int>(value.size()), nullptr, 0,
                                   nullptr, nullptr);
  if (length <= 0) {
    return {};
  }
  std::string result(static_cast<std::size_t>(length), '\0');
  WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
                      result.data(), length, nullptr, nullptr);
  return result;
}

std::string win_error_text(DWORD code) {
  wchar_t* buffer = nullptr;
  DWORD length = FormatMessageW(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      reinterpret_cast<wchar_t*>(&buffer), 0, nullptr);
  std::string message;
  if (length > 0 && buffer != nullptr) {
    message = to_utf8(std::wstring(buffer, length));
    LocalFree(buffer);
  }
  while (!message.empty() &&
         (message.back() == '\r' || message.back() == '\n' || message.back() == ' ')) {
    message.pop_back();
  }
  if (message.empty()) {
    message = "WinHTTP error " + std::to_string(code);
  }
  return message;
}

struct HandleCloser {
  HINTERNET handle = nullptr;
  ~HandleCloser() {
    if (handle != nullptr) {
      WinHttpCloseHandle(handle);
    }
  }
};

HttpResponse winhttp_post(const HttpRequest& request,
                          const StreamCallback& on_chunk,
                          std::atomic<bool>* cancel) {
  HttpResponse response;
  std::wstring url = to_wide(request.url);
  if (url.empty()) {
    response.error = "Invalid URL";
    return response;
  }

  URL_COMPONENTS components{};
  components.dwStructSize = sizeof(components);
  components.dwSchemeLength = static_cast<DWORD>(-1);
  components.dwHostNameLength = static_cast<DWORD>(-1);
  components.dwUrlPathLength = static_cast<DWORD>(-1);
  components.dwExtraInfoLength = static_cast<DWORD>(-1);
  if (!WinHttpCrackUrl(url.c_str(), 0, 0, &components)) {
    response.error = "Invalid URL: " + request.url;
    return response;
  }

  std::wstring host(components.lpszHostName, components.dwHostNameLength);
  std::wstring path(components.lpszUrlPath, components.dwUrlPathLength);
  if (components.dwExtraInfoLength > 0 && components.lpszExtraInfo != nullptr) {
    path += std::wstring(components.lpszExtraInfo, components.dwExtraInfoLength);
  }
  if (path.empty()) {
    path = L"/";
  }
  bool secure = components.nScheme == INTERNET_SCHEME_HTTPS;
  INTERNET_PORT port = components.nPort;

  HandleCloser session;
  session.handle = WinHttpOpen(L"llm-eval-dashboard/1.0",
                               WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                               WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
  if (session.handle == nullptr) {
    session.handle = WinHttpOpen(L"llm-eval-dashboard/1.0",
                                 WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                 WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
  }
  if (session.handle == nullptr) {
    response.error = win_error_text(GetLastError());
    return response;
  }

  int timeout = std::max(1000, request.timeout_ms);
  WinHttpSetTimeouts(session.handle, timeout, timeout, timeout, timeout);

  HandleCloser connection;
  connection.handle = WinHttpConnect(session.handle, host.c_str(), port, 0);
  if (connection.handle == nullptr) {
    response.error = win_error_text(GetLastError());
    return response;
  }

  HandleCloser request_handle;
  request_handle.handle = WinHttpOpenRequest(
      connection.handle, L"POST", path.c_str(), nullptr, WINHTTP_NO_REFERER,
      WINHTTP_DEFAULT_ACCEPT_TYPES, secure ? WINHTTP_FLAG_SECURE : 0);
  if (request_handle.handle == nullptr) {
    response.error = win_error_text(GetLastError());
    return response;
  }

  std::wstring header_block;
  for (const auto& header : request.headers) {
    if (lower_ascii(header.first) == "content-length") {
      continue;
    }
    header_block += to_wide(header.first) + L": " + to_wide(header.second) + L"\r\n";
  }

  if (cancel != nullptr && cancel->load()) {
    response.aborted = true;
    response.error = "cancelled";
    return response;
  }

  BOOL sent = WinHttpSendRequest(
      request_handle.handle,
      header_block.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : header_block.c_str(),
      header_block.empty() ? 0 : static_cast<DWORD>(-1),
      request.body.empty() ? nullptr : const_cast<char*>(request.body.data()),
      static_cast<DWORD>(request.body.size()),
      static_cast<DWORD>(request.body.size()), 0);
  if (!sent) {
    response.error = win_error_text(GetLastError());
    return response;
  }

  if (!WinHttpReceiveResponse(request_handle.handle, nullptr)) {
    response.error = win_error_text(GetLastError());
    return response;
  }

  DWORD status = 0;
  DWORD status_size = sizeof(status);
  WinHttpQueryHeaders(request_handle.handle,
                      WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                      WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_size,
                      WINHTTP_NO_HEADER_INDEX);
  response.status = static_cast<int>(status);

  std::vector<char> buffer(8192);
  for (;;) {
    if (cancel != nullptr && cancel->load()) {
      response.aborted = true;
      response.error = "cancelled";
      return response;
    }
    DWORD bytes_read = 0;
    if (!WinHttpReadData(request_handle.handle, buffer.data(),
                         static_cast<DWORD>(buffer.size()), &bytes_read)) {
      response.error = win_error_text(GetLastError());
      return response;
    }
    if (bytes_read == 0) {
      break;
    }
    if (on_chunk && !on_chunk(buffer.data(), static_cast<std::size_t>(bytes_read))) {
      response.aborted = true;
      response.error = "cancelled";
      return response;
    }
  }
  return response;
}

#else // !_WIN32

struct ParsedUrl {
  std::string scheme;
  std::string host;
  int port = 80;
  std::string path;
};

ParsedUrl parse_url(const std::string& url) {
  ParsedUrl parsed;
  std::string rest = url;
  auto scheme_end = rest.find("://");
  if (scheme_end == std::string::npos) {
    parsed.host = rest;
    parsed.path = "/";
    return parsed;
  }
  parsed.scheme = lower_ascii(rest.substr(0, scheme_end));
  rest = rest.substr(scheme_end + 3);
  auto path_start = rest.find('/');
  std::string authority = rest.substr(0, path_start);
  parsed.path = path_start == std::string::npos ? "/" : rest.substr(path_start);
  auto colon = authority.rfind(':');
  if (colon != std::string::npos && authority.find(']') == std::string::npos) {
    parsed.host = authority.substr(0, colon);
    parsed.port = std::atoi(authority.substr(colon + 1).c_str());
  } else {
    parsed.host = authority;
  }
  if (parsed.port == 0) {
    parsed.port = parsed.scheme == "https" ? 443 : 80;
  }
  return parsed;
}

HttpResponse httplib_post(const HttpRequest& request,
                          const StreamCallback& on_chunk,
                          std::atomic<bool>* cancel) {
  HttpResponse response;
  ParsedUrl parsed = parse_url(request.url);
  if (parsed.scheme == "https") {
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
    response.error = "HTTPS support is not compiled in (install OpenSSL and rebuild)";
    return response;
#else
    httplib::SSLClient client(parsed.host, parsed.port);
    client.enable_server_certificate_verification(true);
    client.set_read_timeout(std::max(1, request.timeout_ms / 1000), 0);
    client.set_connection_timeout(std::max(1, request.timeout_ms / 1000), 0);
    httplib::Headers headers;
    for (const auto& header : request.headers) {
      headers.emplace(header.first, header.second);
    }
    if (cancel != nullptr && cancel->load()) {
      response.aborted = true;
      response.error = "cancelled";
      return response;
    }
    auto result = client.Post(
        parsed.path, headers, request.body, "application/json",
        [&](const char* data, std::size_t length) {
          if (cancel != nullptr && cancel->load()) {
            return false;
          }
          return !on_chunk || on_chunk(data, length);
        });
    if (!result) {
      response.error = httplib::to_string(result.error());
      return response;
    }
    response.status = result->status;
    return response;
#endif
  }

  httplib::Client client(parsed.host, parsed.port);
  client.set_read_timeout(std::max(1, request.timeout_ms / 1000), 0);
  client.set_connection_timeout(std::max(1, request.timeout_ms / 1000), 0);
  httplib::Headers headers;
  for (const auto& header : request.headers) {
    headers.emplace(header.first, header.second);
  }
  if (cancel != nullptr && cancel->load()) {
    response.aborted = true;
    response.error = "cancelled";
    return response;
  }
  auto result = client.Post(
      parsed.path, headers, request.body, "application/json",
      [&](const char* data, std::size_t length) {
        if (cancel != nullptr && cancel->load()) {
          return false;
        }
        return !on_chunk || on_chunk(data, length);
      });
  if (!result) {
    response.error = httplib::to_string(result.error());
    return response;
  }
  response.status = result->status;
  return response;
}

#endif // _WIN32

} // namespace

HttpResponse http_post_stream(const HttpRequest& request,
                              const StreamCallback& on_chunk,
                              std::atomic<bool>* cancel) {
#ifdef _WIN32
  return winhttp_post(request, on_chunk, cancel);
#else
  return httplib_post(request, on_chunk, cancel);
#endif
}

} // namespace llmbench
