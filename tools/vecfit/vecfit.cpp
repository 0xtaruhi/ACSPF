#include <memory>
#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <spdlog/common.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Config.h"
#include "Eval.h"
#include "OrigInfo.h"
#include "TouchStoneParser.h"
#include "utils/Logger.h"
#include "utils/TimeRecorder.h"

#include "OraSolver.h"
#include "VFSolver.h"

int main(int argc, char **argv) {
  logger::init();

  // Parse arguments
  std::shared_ptr<Config> config;
  try {
    config = parseArgs(argc, argv);
  } catch (const std::runtime_error &err) {
    logger::error("Error parsing arguments: {}", err.what());
    return 1;
  }

  if (config->verbose) {
#ifdef SPDLOG_ACTIVE
    logger::set_level(spdlog::level::trace);
#endif
  }

  // Set the number of threads
  if (config->max_threads != -1) {
    omp_set_num_threads(config->max_threads);
  }

  // Parse TouchStone file
  logger::info("Parsing TouchStone file {}...", config->input_file);
  TouchStoneParser parser(config->input_file);
  OrigInfo orig_info;
  try {
    TIME_RECORDER("Parser");
    orig_info = parser.parse();
  } catch (const std::runtime_error &err) {
    logger::error("Error parsing TouchStone file: {}", err.what());
    return 1;
  }
  logger::info("Successfully parsed {} records.", orig_info.freqs.size());

  const auto port_num = orig_info.s_params.dimension(1);
  if (omp_get_max_threads() > port_num * port_num) {
    omp_set_num_threads(port_num * port_num);
  }

  // Solve
  SolverConfig solver_config{config->max_iters, config->pole_num,
                             config->exact_dc, config->reduced_columns};
  // OraSolver solver(orig_info, solver_config);
  std::unique_ptr<Solver> solver;
  if (config->method == "ORA" || config->method == "ora") {
    solver = std::make_unique<OraSolver>(orig_info, solver_config);
  } else if (config->method == "VF" || config->method == "vf") {
    solver = std::make_unique<VFSolver>(orig_info, solver_config);
  } else {
    logger::error("Unknown method: {}", config->method);
    return 1;
  }
  SolveRes *res;
  {
    TIME_RECORDER("Solver");
    res = &solver->solve();
  }

  // Write results to file
  logger::info("Writing results to file {}...", config->output_file);
  {
    TIME_RECORDER("Write to file");
    res->model.writeToFile(config->output_file);
  }
  logger::info("Finished writing results to file.");

  // Evaluate the model
  if (config->do_eval) {
    logger::trace("Start evaluating the model...");

    const auto &orig_h_tensor = solver->getSParams();
    auto fitted_h_tensor = res->model.calHResponse(solver->getOmegas());
    auto eval_res = evalFit(res->model, solver->getOmegas(), orig_h_tensor);
    eval_res.printToLog();

    if (config->write_ref) {
      writeRefFile(orig_h_tensor, fitted_h_tensor, orig_info.freqs, "ref.csv");
      logger::info("Reference file written to ref.csv");
    }
  }

  return 0;
}
