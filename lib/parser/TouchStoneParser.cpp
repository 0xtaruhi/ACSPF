#include <cctype>

#include <Eigen/Core>
#include <cstdlib>
#include <string_view>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "FileMemoryMap.h"
#include "OrigInfo.h"
#include "TouchStoneParser.h"
#include "utils/Logger.h"
#include "utils/Parse.h"

using namespace utils;

namespace {
Eigen::Index parsePortNumFromFileName(std::string_view filename) {
  auto ext = filename.substr(filename.find_last_of('.'));
  auto port_num_str = ext.substr(2, ext.size() - 3);
  return std::atoi(port_num_str.data());
}
} // namespace

TouchStoneParser::TouchStoneParser(std::string_view filename)
    : file_memory_map_(filename) {
  meta_.filename = filename;
  if (!file_memory_map_.getMemory()) {
    logger::error("Failed to open file: {}", filename);
    std::exit(1);
  }
}

const char *TouchStoneParser::parseMeta() {
  auto &meta = meta_;
  const auto &filename = meta.filename;

  meta.num_ports = parsePortNumFromFileName(filename);
  auto file_mem = static_cast<const char *>(file_memory_map_.getMemory());
  meta.freq_unit = TouchStoneMeta::FreqUnit::Hz;

  while (true) {
    skipIfSpaceOrNewLine(file_mem);
    if (unlikely(*file_mem == '\0')) {
      logger::error("Failed to parse TouchStone file: {}", filename);
      std::exit(1);
    }

    if (*file_mem == '!') {
      skipUntilNewLine(file_mem);
      file_mem++;
      continue;
    }

    if (*file_mem == '#') {
      file_mem++;
      skipIfIsSpace(file_mem);
      const auto follow_hz = [](const char *mem) {
        const char c[2] = {mem[0], mem[1]};
        return toupper(c[0]) == 'H' && toupper(c[1]) == 'Z';
      };
      switch (*file_mem) {
      case 'k':
      case 'K':
        meta.freq_unit = TouchStoneMeta::FreqUnit::kHz;
        file_mem++;
        break;
      case 'M':
      case 'm':
        meta.freq_unit = TouchStoneMeta::FreqUnit::MHz;
        file_mem++;
        break;
      case 'G':
      case 'g':
        meta.freq_unit = TouchStoneMeta::FreqUnit::GHz;
        file_mem++;
        break;
      }
      if (follow_hz(file_mem)) {
        file_mem += 2;
      } else {
        logger::error("Failed to parse TouchStone file: {}, invalid freq unit",
                      filename);
        std::exit(1);
      }

      skipIfIsSpace(file_mem);
      if (*file_mem == 'S' || *file_mem == 's') {
        meta.param_type = TouchStoneMeta::ParamType::S;
        file_mem++;
      } else {
        logger::error("Failed to parse TouchStone file: {}, invalid param type",
                      filename);
        std::exit(1);
      }

      skipIfIsSpace(file_mem);

      if (strncmp(file_mem, "DB", 2) == 0) {
        meta.format = TouchStoneMeta::Format::DB;
        file_mem += 2;
      } else if (strncmp(file_mem, "MA", 2) == 0) {
        meta.format = TouchStoneMeta::Format::MA;
        file_mem += 2;
      } else if (strncmp(file_mem, "RI", 2) == 0) {
        meta.format = TouchStoneMeta::Format::RI;
        file_mem += 2;
      } else {
        logger::error("Failed to parse TouchStone file: {}, invalid format",
                      filename);
        std::exit(1);
      }

      skipUntilNewLine(file_mem);
      file_mem++;
      break;
    }
  }

  return file_mem;
}

OrigInfo TouchStoneParser::parse() {
  auto file_mem = parseMeta();
  if (meta_.format == TouchStoneMeta::Format::DB) {
    return parseData<TouchStoneMeta::Format::DB>(file_mem, meta_.freq_unit);
  } else if (meta_.format == TouchStoneMeta::Format::MA) {
    return parseData<TouchStoneMeta::Format::MA>(file_mem, meta_.freq_unit);
  } else if (meta_.format == TouchStoneMeta::Format::RI) {
    return parseData<TouchStoneMeta::Format::RI>(file_mem, meta_.freq_unit);
  } else {
    logger::error("Failed to parse TouchStone file: {}, invalid format",
                  meta_.filename);
    std::exit(1);
  }
}

template <TouchStoneMeta::Format F>
OrigInfo TouchStoneParser::parseData(const char *start_addr,
                                     TouchStoneMeta::FreqUnit freq_unit) {
  // Parse the TouchStone file
  auto *file_memory = start_addr;
  std::vector<std::vector<const char *>> data_ptrs;
  data_ptrs.reserve(16384);

  const auto port_num = meta_.num_ports;

  // Parse the rest of the file
  while (*file_memory != '\0') {
    skipIfIsSpace(file_memory);

    if (unlikely(*file_memory == '!' || *file_memory == '#')) {
      skipUntilNewLine(file_memory);
      file_memory++;
      continue;
    }

    if (unlikely(*file_memory == '\n')) {
      file_memory++;
      continue;
    }

    data_ptrs.push_back(std::vector<const char *>(2 * port_num * port_num + 1));
    auto &data_ptr = data_ptrs.back();

    data_ptr[0] = file_memory;
    skipUntilSpaceOrEof(file_memory);
    skipIfSpaceOrNewLine(file_memory);

    // Parse the S-parameters
    Eigen::MatrixXcd s(port_num, port_num);
    for (Eigen::Index i = 0; i != port_num; ++i) {
      for (Eigen::Index j = 0; j != port_num; ++j) {
        data_ptr[2 * i * port_num + 2 * j + 1] = file_memory;
        skipUntilSpaceOrEof(file_memory);
        skipIfSpaceOrNewLine(file_memory);
        data_ptr[2 * i * port_num + 2 * j + 2] = file_memory;
        skipUntilSpaceOrEof(file_memory);
        skipIfSpaceOrNewLine(file_memory);
      }
    }
  }

  OrigInfo orig_info;
  auto &h_tensor = orig_info.s_params;
  orig_info.freqs.resize(data_ptrs.size());
  auto &freqs = orig_info.freqs;

  h_tensor.resize(data_ptrs.size(), port_num, port_num);

  const auto freq_scale = TouchStoneMeta::freqUnit2Double(freq_unit);
  for (size_t i = 0; i != data_ptrs.size(); ++i) {
    freqs(i) = parseDouble(data_ptrs[i][0]) * freq_scale;
  }

  const auto data_process = [](double data1, double data2) {
    if constexpr (F == TouchStoneMeta::Format::DB) {
      double mag = std::pow(10, data1 / 20);
      double phase = data2 * M_PI / 180;
      data1 = mag * std::cos(phase);
      data2 = mag * std::sin(phase);
    } else if constexpr (F == TouchStoneMeta::Format::MA) {
      double mag = data1;
      double phase = data2 * M_PI / 180;
      data1 = mag * std::cos(phase);
      data2 = mag * std::sin(phase);
    }
    return std::complex<double>(data1, data2);
  };

  if (port_num >= 8) {
    // Use OpenMP for parallelization if the port number is
    // large enough
#pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index q = 0; q != port_num; ++q) {
      for (Eigen::Index m = 0; m != port_num; ++m) {
        for (size_t i = 0; i != data_ptrs.size(); ++i) {
          auto &data_ptr = data_ptrs[i];
          double data1 = parseDouble(data_ptr[2 * q * port_num + 2 * m + 1]);
          double data2 = parseDouble(data_ptr[2 * q * port_num + 2 * m + 2]);
          h_tensor(i, q, m) = data_process(data1, data2);
        }
      }
    }
  } else {
    // Otherwise, use a simple loop
    for (Eigen::Index q = 0; q != port_num; ++q) {
      for (Eigen::Index m = 0; m != port_num; ++m) {
        for (size_t i = 0; i != data_ptrs.size(); ++i) {
          auto &data_ptr = data_ptrs[i];
          double real = parseDouble(data_ptr[2 * q * port_num + 2 * m + 1]);
          double imag = parseDouble(data_ptr[2 * q * port_num + 2 * m + 2]);
          h_tensor(i, q, m) = data_process(real, imag);
        }
      }
    }
  }

  return orig_info;
}
