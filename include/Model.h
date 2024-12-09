#ifndef VECFIT_MODEL_H
#define VECFIT_MODEL_H

#include <Eigen/Core>
#include <string_view>
#include <unsupported/Eigen/CXX11/Tensor>

template <typename OmegasType, typename PolesRType, typename PolesCType,
          typename R0Type, typename ResidualRType, typename ResidualCType>
inline void calHResponse(Eigen::Tensor<std::complex<double>, 3> &h_tensor,
                         const OmegasType &omegas, const PolesRType &poles_real,
                         const PolesCType &poles_complex, const R0Type &r0,
                         const ResidualRType &tensor_Rr,
                         const ResidualCType &tensor_Rc) {
  const auto n_r = poles_real.size();
  const auto n_c = poles_complex.size();

  const auto k_bar = omegas.size();
  const auto q_bar = r0.rows();
  const auto m_bar = r0.cols();

  constexpr auto kImagUnit = std::complex<double>(0.0, 1.0);

  h_tensor = Eigen::Tensor<std::complex<double>, 3>(k_bar, q_bar, m_bar);

  Eigen::MatrixXcd mat_phi_real = Eigen::MatrixXcd(k_bar, n_r);
#pragma omp parallel for
  for (Eigen::Index col = 0; col != poles_real.size(); ++col) {
    mat_phi_real.col(col).noalias() =
        (1.0 / (kImagUnit * omegas.array() - poles_real(col))).matrix();
  }

  Eigen::MatrixXcd mat_phi_complex = Eigen::MatrixXcd(k_bar, 2 * n_c);

#pragma omp parallel for
  for (Eigen::Index i = 0; i != n_c; ++i) {
    Eigen::VectorXcd m = 1.0 / (kImagUnit * omegas.array() - poles_complex(i));
    Eigen::VectorXcd n =
        1.0 / (kImagUnit * omegas.array() - std::conj(poles_complex(i)));
    mat_phi_complex.col(2 * i).noalias() = m + n;
    mat_phi_complex.col(2 * i + 1).noalias() = kImagUnit * (m - n);
  }

#pragma omp parallel for collapse(2)
  for (Eigen::Index q = 0; q != q_bar; ++q) {
    for (Eigen::Index m = 0; m != m_bar; ++m) {
      auto h_tensor_qm = Eigen::Map<Eigen::VectorXcd>(
          h_tensor.data() + m * q_bar * k_bar + q * k_bar, k_bar);
      const auto tensor_Rr_qm = Eigen::Map<const Eigen::VectorXd>(
          tensor_Rr.data() + m * q_bar * n_r + q * n_r, n_r);
      const auto tensor_Rc_qm = Eigen::Map<const Eigen::VectorXcd>(
          tensor_Rc.data() + m * q_bar * n_c + q * n_c, n_c);
      Eigen::VectorXd extended_Rc_qm = Eigen::VectorXd(2 * n_c);
      for (Eigen::Index i = 0; i != n_c; ++i) {
        extended_Rc_qm(2 * i) = tensor_Rc_qm(i).real();
        extended_Rc_qm(2 * i + 1) = tensor_Rc_qm(i).imag();
      }

      h_tensor_qm =
          (mat_phi_real * tensor_Rr_qm + mat_phi_complex * extended_Rc_qm)
              .array() +
          r0(q, m);
    }
  }
}

struct Model {
  Model() = default;
  Model(std::string_view filename);

  Eigen::VectorXd poles_real;
  Eigen::VectorXcd poles_complex;
  Eigen::MatrixXd r0;
  Eigen::Tensor<double, 3> tensor_Rr;
  Eigen::Tensor<std::complex<double>, 3> tensor_Rc;

  Eigen::Tensor<std::complex<double>, 3> &
  calHResponse(Eigen::Tensor<std::complex<double>, 3> &h_tensor,
               const Eigen::VectorXd &omegas) const {
    ::calHResponse(h_tensor, omegas, poles_real, poles_complex, r0, tensor_Rr,
                   tensor_Rc);
    return h_tensor;
  }

  auto calHResponse(const Eigen::VectorXd &omegas) const {
    Eigen::Tensor<std::complex<double>, 3> h_tensor;
    ::calHResponse(h_tensor, omegas, poles_real, poles_complex, r0, tensor_Rr,
                   tensor_Rc);
    return h_tensor;
  }

  auto getPolesNum() const noexcept {
    return 2 * poles_complex.size() + poles_real.size();
  }

  void writeToFile(std::string_view filename);
};

#endif // VECFIT_MODEL_H
