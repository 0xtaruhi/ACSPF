#ifndef VECFIT_VF_SOLVER_H
#define VECFIT_VF_SOLVER_H

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include "OrigInfo.h"
#include "Solver.h"

class VFSolver : public Solver {
public:
  VFSolver(const Eigen::VectorXd &freqs,
           const Eigen::Tensor<std::complex<double>, 3> &h_tensor,
           SolverConfig solver_config = kDefaultSolverConfig);

  VFSolver(const OrigInfo &orig_info,
           SolverConfig config = kDefaultSolverConfig)
      : VFSolver(orig_info.freqs, orig_info.s_params, config) {}

  SolveRes &solve() noexcept override;

private:
  void solveWithFixedPolesNum(int poles_num) noexcept;

  void updatePhiMatrices(const Eigen::VectorXd &real_poles,
                         const Eigen::VectorXcd &complex_poles) noexcept;
  Eigen::MatrixXcd phi_real_;
  Eigen::MatrixXcd phi_complex_;

  Eigen::MatrixX<Eigen::VectorXd> mat_Sqm_realimag_;
};

#endif // VECFIT_VF_SOLVER_H
