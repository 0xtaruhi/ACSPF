#ifndef VECFIT_TOUCH_STONE_PARSER_H
#define VECFIT_TOUCH_STONE_PARSER_H

#include <string_view>

#include "FileMemoryMap.h"
#include "OrigInfo.h"

struct TouchStoneMeta {
  enum class FreqUnit {
    Hz,
    kHz,
    MHz,
    GHz,
  };

  static double freqUnit2Double(FreqUnit unit) {
    switch (unit) {
    case FreqUnit::Hz:
      return 1;
    case FreqUnit::kHz:
      return 1e3;
    case FreqUnit::MHz:
      return 1e6;
    case FreqUnit::GHz:
      return 1e9;
    }
    return 1;
  }

  enum class ParamType {
    S,
  };

  enum class Format {
    DB,
    MA,
    RI,
  };

  std::string filename;
  FreqUnit freq_unit;
  ParamType param_type;
  Format format;
  Eigen::Index num_ports;
};
class TouchStoneParser {
public:
  TouchStoneParser(std::string_view filename);

  OrigInfo parse();

  TouchStoneMeta meta() const { return meta_; }

private:
  const char *parseMeta();

  template <TouchStoneMeta::Format F>
  OrigInfo parseData(const char *start_addr, TouchStoneMeta::FreqUnit freq_unit);

private:
  TouchStoneMeta meta_;
  FileMemoryMap file_memory_map_;
};

#endif // VECFIT_TOUCH_STONE_PARSER_H
