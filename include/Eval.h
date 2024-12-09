#ifndef VECFIT_EVAL_H
#define VECFIT_EVAL_H

#include <complex>

#include <Eigen/Core>
#include <string_view>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Model.h"

struct EvalRes {
  Eigen::Tensor<std::complex<double>, 3> fitted_h_tensor;

  double err;
  double kerr;
  double freq0_err;
  double svd_r0;

  void printToLog() const;
};

EvalRes evalFit(const Model &model, const Eigen::VectorXd &omegas,
                const Eigen::Tensor<std::complex<double>, 3> &orig_h_tensor);

double getErr(const Eigen::Tensor<std::complex<double>, 3> &orig_h_tensor,
              const Eigen::Tensor<std::complex<double>, 3> &fitted_h_tensor);

void writeRefFile(const Eigen::Tensor<std::complex<double>, 3> &orig_h_tensor,
                  const Eigen::Tensor<std::complex<double>, 3> &fitted_h_tensor,
                  const Eigen::ArrayXd &freqs, std::string_view filename);

#endif // VECFIT_EVAL_H
