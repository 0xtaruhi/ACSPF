#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Config.h"
#include "Eval.h"
#include "Model.h"
#include "OrigInfo.h"
#include "TouchStoneParser.h"
#include "utils/Logger.h"

int main(int argc, char **argv) {
  EvaltoolConfig config;
  try {
    config = parseArgs(argc, argv);
  } catch (const std::exception &e) {
    logger::error("{}", e.what());
    return 1;
  }

  if (config.verbose) {
    logger::set_level(spdlog::level::trace);
  }
  logger::trace("Loading model and touchstone file...");

  Model model(config.model_file);
  logger::trace("Model loaded from file: {}", config.model_file);

  OrigInfo orig_info;

  try {
    orig_info = TouchStoneParser(config.touchstone_file).parse();
  } catch (const std::runtime_error &err) {
    logger::error("Error parsing TouchStone file: {}", err.what());
    return 1;
  }
  logger::trace("Touchstone file loaded: {}", config.touchstone_file);

  const auto &freqs = orig_info.freqs;
  const auto omegas = freqs * 2 * M_PI;

  logger::trace("Start evaluating the model...");
  auto eval_res = evalFit(model, omegas, orig_info.s_params);
  eval_res.printToLog();
  if (config.write_ref) {
    writeRefFile(orig_info.s_params, eval_res.fitted_h_tensor, freqs,
                 "ref.csv");
    logger::info("Reference File written to ref.csv");
  }
  return 0;
}
