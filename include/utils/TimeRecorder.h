#ifndef TIME_RECORDER_H
#define TIME_RECORDER_H

#include "Logger.h"
#include <chrono>
#include <string_view>

#ifdef TIME_MEASURE
class TimeRecorder {
public:
  TimeRecorder(std::string_view name) noexcept
      : start_(std::chrono::high_resolution_clock::now()), name_(name) {}

  ~TimeRecorder() noexcept {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
    logger::trace("{} finished in {} ms.", name_, duration.count());
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::string_view name_;
};
#else
class TimeRecorder {
public:
  TimeRecorder(std::string_view) noexcept {}

  ~TimeRecorder() noexcept = default;
};

#endif // TIME_MEASURE

#define TIME_RECORDER(name) TimeRecorder time_recorder(name)

#endif // TIME_RECORDER_H
