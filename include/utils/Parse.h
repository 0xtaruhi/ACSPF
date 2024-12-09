#ifndef VECFIT_UTILS_PARSE_H
#define VECFIT_UTILS_PARSE_H

#include <cstddef>

#include "fast_double_parser.h"

#define ALWAYS_INLINE __attribute__((always_inline)) inline

namespace utils {
template <char... Chars>
ALWAYS_INLINE size_t skipIfIsOneOf(const char *&file_memory) {
  const char *start = file_memory;
  while (((*file_memory == Chars) || ...)) {
    ++file_memory;
  }
  return file_memory - start;
}

template <char... Chars>
ALWAYS_INLINE size_t skipIfIsNotOneOf(const char *&file_memory) {
  const char *start = file_memory;
  while (((*file_memory != Chars) && ...)) {
    ++file_memory;
  }
  return file_memory - start;
}

ALWAYS_INLINE size_t skipIfIsSpace(const char *&file_memory) {
  return skipIfIsOneOf<' ', '\t', '\r'>(file_memory);
}

ALWAYS_INLINE size_t skipIfSpaceOrNewLine(const char *&file_memory) {
  return skipIfIsOneOf<' ', '\t', '\n', '\r'>(file_memory);
}

ALWAYS_INLINE size_t skipUntilNewLine(const char *&file_memory) {
  return skipIfIsNotOneOf<'\n'>(file_memory);
}

ALWAYS_INLINE size_t skipUntilSpaceOrEof(const char *&file_memory) {
  return skipIfIsNotOneOf<' ', '\t', '\n', '\r', '\0'>(file_memory);
}

ALWAYS_INLINE double parseDouble(const char *&file_memory) {
  double res = 0;
  auto end = fast_double_parser::parse_number(file_memory, &res);
  file_memory = end;
  return res;
}

} // namespace utils

#undef ALWAYS_INLINE

#endif // VECFIT_UTILS_PARSE_H
