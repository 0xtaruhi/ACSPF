#ifndef VECFIT_SOLVER_H
#define VECFIT_SOLVER_H

#include <Eigen/Core>
#include <cmath>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Model.h"
#include "OrigInfo.h"
#include "utils/Logger.h"

struct SolveRes {
  Model model;
};

struct SolverConfig {
  int max_iters;
  int pole_num;
  bool exact_dc;
  bool reduced_columns;
};

static constexpr SolverConfig kDefaultSolverConfig =
    SolverConfig{30, -1, true, true};

class Solver {
public:
  Solver(const Eigen::VectorXd &freqs,
         const Eigen::Tensor<std::complex<double>, 3> &s_params,
         SolverConfig config = kDefaultSolverConfig)
      : omegas_(freqs * 2 * M_PI), s_params_(s_params),
        config_(std::move(config)) {
    ndata_ = getPortNum() * getPortNum();
  }

  Solver(const OrigInfo &orig_info, SolverConfig config = kDefaultSolverConfig)
      : Solver(orig_info.freqs, orig_info.s_params, config) {}

  virtual ~Solver() {}

  virtual SolveRes &solve() noexcept = 0;

  const auto &getSParams() const noexcept { return s_params_; }
  const auto &getOmegas() const noexcept { return omegas_; }

protected:
  Eigen::Index getFreqNum() const noexcept { return s_params_.dimension(0); }
  Eigen::Index getPortNum() const noexcept { return s_params_.dimension(1); }
  Eigen::Index getReducedDataNum() const noexcept { return ndata_; }

  Eigen::Index reduceColumns(double target_err = 1e-5) noexcept;

  Eigen::Index predictPolesNum() const noexcept;

  Eigen::Index getPoleNum() const noexcept {
    Eigen::Index pole_num;
    if (config_.pole_num == -1) {
      pole_num = predictPolesNum();
      logger::trace("Predicted poles number: {}", pole_num);
    } else {
      pole_num = config_.pole_num;
      logger::trace("Fixed poles number: {}", pole_num);
    }
    return pole_num;
  }

  Eigen::Map<const Eigen::VectorXcd>
  getSqmVectorMap(Eigen::Index q, Eigen::Index m) const noexcept {
    const auto n = m * getPortNum() + q;
    return getSqmVectorMap(n);
  }

  Eigen::Map<const Eigen::VectorXcd>
  getSqmVectorMap(Eigen::Index n) const noexcept {
    return Eigen::Map<const Eigen::VectorXcd>(
        s_params_.data() + n * getFreqNum(), getFreqNum());
  }

  void calculatePhiMatrices(const Eigen::VectorXd &real_poles,
                            const Eigen::VectorXcd &complex_poles,
                            Eigen::MatrixXcd &phi_real,
                            Eigen::MatrixXcd &phi_complex) const noexcept;

protected:
  Eigen::VectorXd omegas_;
  Eigen::Index ndata_;
  const Eigen::Tensor<std::complex<double>, 3> &s_params_;
  SolveRes solve_res_;
  SolverConfig config_;
};

#endif // VECFIT_SOLVER_H
