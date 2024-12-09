#ifndef VECFIT_LOGGER_H
#define VECFIT_LOGGER_H

#include <cstdint> // IWYU pragma: export

#ifdef SPDLOG_ACTIVE
#include <spdlog/spdlog.h>
#else
#include <cstdio>
#endif

namespace logger {

#ifdef NDEBUG
#define DEFAULT_LOG_LEVEL INFO
#else
#define DEFAULT_LOG_LEVEL DEBUG
#endif

#ifdef SPDLOG_ACTIVE

inline void set_level(spdlog::level::level_enum level) {
  spdlog::set_level(level);
}

inline void init() {
  spdlog::set_pattern("[%H:%M:%S:%e] [%^%l%$] %v");
#ifdef NDEBUG
  set_level(spdlog::level::info);
#else
  set_level(spdlog::level::debug);
#endif
}

template <typename... Args>
inline void trace(spdlog::format_string_t<Args...> fmt, Args &&...args) {
  spdlog::trace(fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void debug(spdlog::format_string_t<Args...> fmt, Args &&...args) {
  spdlog::debug(fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void info(spdlog::format_string_t<Args...> fmt, Args &&...args) {
  spdlog::info(fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void warn(spdlog::format_string_t<Args...> fmt, Args &&...args) {
  spdlog::warn(fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void error(spdlog::format_string_t<Args...> fmt, Args &&...args) {
  spdlog::error(fmt, std::forward<Args>(args)...);
}

#else

inline int log_level;
enum LogLevel { TRACE = 0, DEBUG, INFO, WARN, ERROR };

inline void init() { log_level = DEFAULT_LOG_LEVEL; }

inline void set_level(int level) { log_level = level; }

template <typename... Args>
inline void trace(const char *fmt, const Args &...args) {
  if (log_level <= TRACE) {
    std::printf(fmt, args...);
  }
}

template <typename... Args>
inline void debug(const char *fmt, const Args &...args) {
  if (log_level <= DEBUG) {
    std::printf(fmt, args...);
  }
}

template <typename... Args>
inline void info(const char *fmt, const Args &...args) {
  if (log_level <= INFO) {
    std::printf(fmt, args...);
  }
}

template <typename... Args>
inline void warn(const char *fmt, const Args &...args) {
  if (log_level <= WARN) {
    std::printf(fmt, args...);
  }
}

template <typename... Args>
inline void error(const char *fmt, const Args &...args) {
  if (log_level <= ERROR) {
    std::printf(fmt, args...);
  }
}

#endif

} // namespace logger

#endif // VECFIT_LOGGER_H
