#ifndef EVALTOOL_CONFIG_H
#define EVALTOOL_CONFIG_H

#include <string>

struct EvaltoolConfig {
  std::string model_file;
  std::string touchstone_file;
  bool write_ref = false;
  bool verbose = false;
};

EvaltoolConfig parseArgs(int argc, char **argv);

#endif // EVALTOOL_CONFIG_H
