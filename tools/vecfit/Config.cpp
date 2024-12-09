#include "Config.h"

#include <Eigen/src/Core/util/Macros.h>
#include <argparse.hpp>
#include <filesystem>
#include <fmt/format.h>
#include <sstream>

namespace {
void printVersion() {
  std::stringstream ss;
  ss << "Version " << PROJECT_VERSION << '(' << GIT_COMMIT_HASH << ')'
     << std::endl;
  ss << "Author: " << "Zhengyi Zhang & Sijing Yang (Fudan University)"
     << std::endl;
  ss << "Built with: " << std::endl;
  ss << "- Eigen " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "."
     << EIGEN_MINOR_VERSION << std::endl;
  ss << "- OpenMP " << _OPENMP << std::endl;
  ss << "- C++ " << __cplusplus << std::endl;
  ss << "- Compiler: " << __VERSION__ << std::endl;
  ss << "Built on: " << __DATE__ << " " << __TIME__ << std::endl;

  std::cout << ss.str();
  std::exit(0);
}
} // namespace

#define DEFUALT_OUTPUT_FILENAME (DEFAULT_OUTPUT_SUFFIX "_{case}.dat")

std::shared_ptr<Config> parseArgs(int argc, char **argv) {
  argparse::ArgumentParser program("vecfit", PROJECT_VERSION,
                                   argparse::default_arguments::help);

  program.add_argument("-m", "--method")
      .help("method to use, ORA or VF")
      .default_value("ORA")
      .action([](const std::string &value) { return value; });
  program.add_argument("-p", "--pole")
      .help("number of poles, -1 for auto")
      .default_value(-1)
      .action([](const std::string &value) { return std::stoi(value); });
  program.add_argument("-e", "--eval")
      .help("evaluate the model after fitting")
      .flag();
  program.add_argument("--write-ref")
      .help("write reference file, only available when --eval is set")
      .flag();
  program.add_argument("--max-iters")
      .help("maximum number of iterations, default 35 for VF, 8 for ORA")
      .action([](const std::string &value) { return std::stoi(value); });
  program.add_argument("--no-exact-dc").help("do not use exact DC").flag();
  program.add_argument("--no-reduced-columns")
      .help("use all data while pole fitting")
      .flag();
  program.add_argument("-t", "--threads")
      .help("maximum number of threads, -1 for hardware concurrency")
      .default_value(8)
      .action([](const std::string &value) { return std::stoi(value); });
  program.add_argument("-v", "--verbose")
      .help("print verbose information")
      .flag();
  program.add_argument("-V", "--version")
      .help("print version")
      .flag()
      .nargs(0)
      .action([](const auto &) {
        printVersion();
        std::exit(0);
      });
  program.add_argument("file").help("input file, TouchStone format").required();
  program.add_argument("output")
      .help("output file")
      .default_value(DEFUALT_OUTPUT_FILENAME);

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    throw std::runtime_error(err.what());
  }

  auto config = std::make_shared<Config>();
  config->input_file = program.get<std::string>("file");
  if (program.get<std::string>("output") == DEFUALT_OUTPUT_FILENAME) {
    std::filesystem::path path = config->input_file;
    config->output_file =
        fmt::format("{}_{}.dat", DEFAULT_OUTPUT_SUFFIX, path.stem().string());
  } else {
    config->output_file = program.get<std::string>("output");
  }
  config->method = program.get<std::string>("--method");
  if (!program.is_used("--max-iters")) {
    if (config->method == "ORA" || config->method == "ora") {
      config->max_iters = 35;
    } else {
      config->max_iters = 8;
    }
  } else {
    config->max_iters = program.get<int>("--max-iters");
  }
  config->pole_num = program.get<int>("--pole");
  config->max_threads = program.get<int>("--threads");
  config->exact_dc = !program.get<bool>("--no-exact-dc");
  config->reduced_columns = !program.get<bool>("--no-reduced-columns");
  config->do_eval = program.get<bool>("--eval");
  config->write_ref = program.get<bool>("--write-ref");
  config->verbose = program.get<bool>("--verbose");

  if (config->write_ref && !config->do_eval) {
    throw std::runtime_error(
        "--write-ref is only available when --eval is set");
  }

  return config;
}
