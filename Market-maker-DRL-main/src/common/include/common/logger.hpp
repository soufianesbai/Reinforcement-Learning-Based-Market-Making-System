#pragma once

#include <iostream>
#include <source_location>
#include <string_view>
#include <syncstream>

#include "common/timestamp.hpp"

namespace hft::common {

enum class LogLevel : uint8_t { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3, CRITICAL = 4 };

class Logger {
  public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void set_level(LogLevel level) {
        min_level_ = level;
    }

    void log(LogLevel level,
             std::string_view message,
             const std::source_location& location = std::source_location::current()) {
        if (level < min_level_) {
            return;
        }

        std::osyncstream(std::cout) << "[" << to_nanos(now()) << "] "
                                    << "[" << level_to_string(level) << "] "
                                    << "[" << location.file_name() << ":"
                                    << location.line() << "] " << message << "\n";
    }

  private:
    Logger() = default;
    LogLevel min_level_ = LogLevel::INFO;

    static constexpr std::string_view level_to_string(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG:
                return "DEBUG";
            case LogLevel::INFO:
                return "INFO";
            case LogLevel::WARNING:
                return "WARNING";
            case LogLevel::ERROR:
                return "ERROR";
            case LogLevel::CRITICAL:
                return "CRITICAL";
        }
        return "UNKNOWN";
    }
};

// Convenience macros
#define LOG_DEBUG(msg) hft::common::Logger::instance().log(hft::common::LogLevel::DEBUG, msg)
#define LOG_INFO(msg) hft::common::Logger::instance().log(hft::common::LogLevel::INFO, msg)
#define LOG_WARNING(msg) hft::common::Logger::instance().log(hft::common::LogLevel::WARNING, msg)
#define LOG_ERROR(msg) hft::common::Logger::instance().log(hft::common::LogLevel::ERROR, msg)
#define LOG_CRITICAL(msg) \
    hft::common::Logger::instance().log(hft::common::LogLevel::CRITICAL, msg)

}  // namespace hft::common
