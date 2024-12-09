#ifndef VECFIT_CONFIG_H
#define VECFIT_CONFIG_H

#include <memory>
#include <string>

struct Config : public std::enable_shared_from_this<Config> {
  std::shared_ptr<Config> getSharedPtr() { return shared_from_this(); }

  std::string input_file;
  std::string output_file = "output.txt";
  std::string method;
  int max_iters;
  int pole_num;
  int max_threads;
  bool exact_dc;
  bool reduced_columns;
  bool do_eval;
  bool write_ref;
  bool verbose;
};

std::shared_ptr<Config> parseArgs(int argc, char **argv);

#endif // VECFIT_CONFIG_H
