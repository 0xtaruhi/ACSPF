#include "FileMemoryMap.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

FileMemoryMap::FileMemoryMap(std::string_view file_name) noexcept {
  int fd = open(file_name.data(), O_RDONLY);
  if (fd == -1) {
    return;
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    return;
  }

  file_size_ = sb.st_size;
  file_memory_map_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);

  if (file_memory_map_ == MAP_FAILED) {
    file_memory_map_ = nullptr;
    return;
  }

  file_memory_ = file_memory_map_;
}

FileMemoryMap::~FileMemoryMap() {
  if (file_memory_map_) {
    munmap(file_memory_map_, file_size_);
  }
}
