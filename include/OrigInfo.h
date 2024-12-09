#ifndef VECFIT_ORIG_INFO_H
#define VECFIT_ORIG_INFO_H

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

struct OrigInfo {
  Eigen::VectorXd freqs;
  Eigen::Tensor<std::complex<double>, 3> s_params;
};

#endif // VECFIT_ORIG_INFO_H
