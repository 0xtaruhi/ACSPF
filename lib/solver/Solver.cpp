#include <Eigen/Eigen>

#include "Solver.h"

namespace {
static constexpr std::complex<double> kImagUnit(0.0, 1.0);

void calPhiMatricesReal(Eigen::MatrixXcd &phi_real,
                        const Eigen::VectorXd &omegas,
                        const Eigen::VectorXd &poles_real) {
  phi_real.resize(omegas.size(), poles_real.size());

#pragma omp parallel for
  for (Eigen::Index col = 0; col != poles_real.size(); ++col) {
    phi_real.col(col).noalias() =
        (1.0 / (kImagUnit * omegas.array() - poles_real(col))).matrix();
  }
}

void calPhiMatricesComplex(Eigen::MatrixXcd &phi_complex,
                           const Eigen::VectorXd &omegas,
                           const Eigen::VectorXcd &poles) {
  phi_complex.resize(omegas.size(), poles.size() * 2);

#pragma omp parallel for
  for (Eigen::Index col = 0; col != poles.size(); ++col) {
    Eigen::VectorXcd m = 1.0 / (kImagUnit * omegas.array() - poles(col));
    Eigen::VectorXcd n =
        1.0 / (kImagUnit * omegas.array() - std::conj(poles(col)));

    phi_complex.col(col * 2).noalias() = m + n;
    phi_complex.col(col * 2 + 1).noalias() = kImagUnit * (m - n);
  }
}
} // namespace

void Solver::calculatePhiMatrices(
    const Eigen::VectorXd &real_poles, const Eigen::VectorXcd &complex_poles,
    Eigen::MatrixXcd &phi_real, Eigen::MatrixXcd &phi_complex) const noexcept {
#pragma omp parallel sections
  {
#pragma omp section
    { calPhiMatricesReal(phi_real, omegas_, real_poles); }
#pragma omp section
    { calPhiMatricesComplex(phi_complex, omegas_, complex_poles); }
  }
}

Eigen::Index Solver::reduceColumns(double target_err) noexcept {
  const auto k_bar = getFreqNum();
  const auto q_bar = getPortNum();

  const auto n_data = q_bar * q_bar;
  if (n_data < 8) {
    return n_data;
  }
  // Calculate normalized H tensor
  Eigen::MatrixXd norm_factor = Eigen::MatrixXd(q_bar, q_bar);
  Eigen::Tensor<double, 3> normalized_h_tensor(2 * k_bar, q_bar, q_bar);
#pragma omp parallel for collapse(2)
  for (Eigen::Index q = 0; q != q_bar; ++q) {
    for (Eigen::Index m = 0; m != q_bar; ++m) {
      const auto &h_qm = getSqmVectorMap(q, m);
      double norm = h_qm.norm();
      const auto real_offset = m * q_bar * 2 * k_bar + q * 2 * k_bar;
      const auto imag_offset = real_offset + k_bar;
      norm_factor(q, m) = norm;
      Eigen::Map<Eigen::VectorXd> real_part(
          normalized_h_tensor.data() + real_offset, k_bar);
      Eigen::Map<Eigen::VectorXd> imag_part(
          normalized_h_tensor.data() + imag_offset, k_bar);
      real_part = h_qm.real() / norm;
      imag_part = h_qm.imag() / norm;
    }
  }

  const auto get_reduced_columns = [target_err](auto &&svd) {
    Eigen::Index reduced_columns = svd.singularValues().size();
    double threshold = target_err * svd.singularValues().sum();
    double sum = 0;
    for (Eigen::Index i = svd.singularValues().size() - 1; i >= 0; --i) {
      sum += svd.singularValues()(i);
      if (sum >= threshold) {
        reduced_columns = i + 1;
        break;
      }
    }
    return reduced_columns;
  };

  const auto mat_map = Eigen::Map<Eigen::MatrixXd>(normalized_h_tensor.data(),
                                                   2 * k_bar, q_bar * q_bar);
  Eigen::Index reduced_columns;

  if (q_bar * q_bar > 10 * k_bar) {
    Eigen::MatrixXd mat_selfadjoint = mat_map * mat_map.transpose();
    Eigen::BDCSVD<Eigen::MatrixXd> svd(
        mat_selfadjoint, Eigen::ComputeThinU | Eigen::ComputeThinV);
    reduced_columns = 3 * get_reduced_columns(svd);
  } else {
    Eigen::BDCSVD<Eigen::MatrixXd> svd(mat_map, Eigen::ComputeThinU |
                                                    Eigen::ComputeThinV);
    reduced_columns = get_reduced_columns(svd);
  }

  return reduced_columns;
}

Eigen::Index Solver::predictPolesNum() const noexcept {
  const auto findMonotonicIntervalsNum = [](auto &&v, auto &&cmp) {
    enum class MonotonicType { kNone, kIncreasing, kDecreasing };
    MonotonicType type = MonotonicType::kNone;

    Eigen::Index num = 0;
    for (Eigen::Index i = 1; i != v.size(); ++i) {
      if (cmp(v(i - 1), v(i))) {
        if (type == MonotonicType::kDecreasing) {
          ++num;
        }
        type = MonotonicType::kIncreasing;
      } else {
        if (type == MonotonicType::kIncreasing) {
          ++num;
        }
      }
    }
    return num + 1;
  };

  Eigen::VectorXi mono_nums(ndata_);

#pragma omp parallel for
  for (Eigen::Index i = 0; i != ndata_; ++i) {
    const auto &data = getSqmVectorMap(i);
    const auto mono_num = findMonotonicIntervalsNum(
        data, [](auto &&a, auto &&b) { return std::abs(a) < std::abs(b); });
    mono_nums(i) = mono_num;
  }

  const auto mean = mono_nums.mean();
  auto ret = mean;
  if (mean > 300) {
    ret = 20 * std::log2(mean);
  } else if (mean < 6) {
    ret = 6;
  }
  return ret;
}
