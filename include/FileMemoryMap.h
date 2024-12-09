#ifndef VECFIT_FILE_MEMORY_MAP_H
#define VECFIT_FILE_MEMORY_MAP_H

#include <string_view>
class FileMemoryMap {
public:
  FileMemoryMap(std::string_view file_name) noexcept;

  ~FileMemoryMap();

  void *getMemory() const noexcept { return file_memory_; }

  size_t getSize() const noexcept { return file_size_; }

private:
  void *file_memory_map_;
  void *file_memory_;
  size_t file_size_;
};

#endif // VECFIT_FILE_MEMORY_MAP_H
