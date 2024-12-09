#ifndef VECFIT_ORA_SOLVER_H
#define VECFIT_ORA_SOLVER_H

#include "Solver.h"

class OraSolver : public Solver {
public:
  using Solver::Solver;

  SolveRes &solve() noexcept override;

private:
  auto oraSolve(Eigen::Index n, Eigen::Index d) -> std::pair<double, Eigen::VectorXcd>;
  void numfit(Eigen::Index n);
  void denfit(Eigen::Index n, Eigen::Index d);

  void calculateModel(const Eigen::VectorXcd& poles) noexcept;

  Eigen::VectorXcd den_;
  Eigen::VectorXcd poles_;
  Eigen::MatrixXd mat_Q_;
  Eigen::MatrixXd mat_H_;
};

#endif // VECFIT_ORA_SOLVER
