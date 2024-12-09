#include "Config.h"

#include <argparse.hpp>

EvaltoolConfig parseArgs(int argc, char **argv) {
  argparse::ArgumentParser program("evaltool", PROJECT_VERSION);

  // Add arguments
  program.add_argument("model_file").help("model file");
  program.add_argument("touchstone_file").help("touchstone file");
  program.add_argument("--write-ref")
      .help("write reference file")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("-v")
      .help("verbose")
      .default_value(false)
      .implicit_value(true);

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    throw std::runtime_error(err.what());
  }

  EvaltoolConfig config;
  config.model_file = program.get<std::string>("model_file");
  config.touchstone_file = program.get<std::string>("touchstone_file");
  config.write_ref = program.get<bool>("--write-ref");
  config.verbose = program.get<bool>("-v");

  return config;
}
