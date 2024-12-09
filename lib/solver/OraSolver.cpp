#include "OraSolver.h"
#include "utils/Logger.h"
#include <Eigen/Eigen>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/QR/HouseholderQR.h>

static constexpr std::complex<double> kImagUnit(0.0, 1.0);

SolveRes &OraSolver::solve() noexcept {
  if (config_.reduced_columns) {
    ndata_ = reduceColumns();
    logger::trace("Reduced columns: {}", ndata_);
  } else {
    ndata_ = getPortNum() * getPortNum();
  }

  auto pole_num = getPoleNum();
  if (config_.pole_num == -1) {
    Eigen::Index left = pole_num / 3;
    Eigen::Index right = pole_num * 1.2;
    const auto [default_err, default_poles] = oraSolve(right, right);
    auto best_kerr = default_err * right;
    int iters = 0;
    while (iters != 3 && left < right) {
      Eigen::Index mid = (left + right) / 2;
      const auto [err, best_poles] = oraSolve(mid, mid);
      auto kerr = err * mid;
      logger::trace("Attempt pole num: {}, err: {}, kerr: {}", mid, err, kerr);
      if (kerr < best_kerr) {
        right = mid;
        best_kerr = kerr;
      } else {
        left = mid + 1;
      }
      iters++;
    }
    pole_num = (left + right) / 2;
    logger::trace("Best pole num: {}", pole_num);
    const auto [err, best_poles] = oraSolve(pole_num, pole_num);
    calculateModel(best_poles);
  } else {
    const auto [err, best_poles] = oraSolve(config_.pole_num, config_.pole_num);
    calculateModel(best_poles);
  }
  return solve_res_;
}

auto OraSolver::oraSolve(Eigen::Index n, Eigen::Index d)
    -> std::pair<double, Eigen::VectorXcd> {
  den_ = Eigen::VectorXcd::Ones(getFreqNum());
  mat_Q_ = Eigen::MatrixXd(2 * getFreqNum(), n + 1);
  mat_H_ = Eigen::MatrixXd::Zero(n + 1, n);

  Eigen::MatrixXd f_ri_reduced(2 * getFreqNum(), ndata_);
  Eigen::VectorXcd best_poles;
#pragma omp parallel for
  for (Eigen::Index i = 0; i != ndata_; ++i) {
    f_ri_reduced.block(0, i, getFreqNum(), 1) = getSqmVectorMap(i).real();
    f_ri_reduced.block(getFreqNum(), i, getFreqNum(), 1) =
        getSqmVectorMap(i).imag();
  }

  int stable_cnt = 0;
  double last_err = 1;
  double best_err = 1;
  int iter_num = 0;

  for (; iter_num != 3;) {
    iter_num++;
    denfit(n, d);
    numfit(n);
  }

  while (iter_num != config_.max_iters) {
    iter_num++;
    denfit(n, d);
    numfit(n);

    // Error estimation
    Eigen::MatrixXd g = mat_Q_.householderQr().solve(f_ri_reduced);
    Eigen::MatrixXd fit_ri = mat_Q_ * g;
    double err_n = 0;
    double err_d = 0;
#pragma omp parallel for reduction(+ : err_n, err_d)
    for (Eigen::Index j = 0; j != ndata_; ++j) {
      err_n += (fit_ri.col(j) - f_ri_reduced.col(j)).norm();
      err_d += f_ri_reduced.col(j).norm();
    }
    double err = err_n / err_d;
    if (err < best_err) {
      best_err = err;
      best_poles = poles_;
    }
    if (err < 0.1 && err < 1.2 * last_err && err > 0.8 * last_err) {
      stable_cnt++;
    } else {
      stable_cnt = 0;
    }
    last_err = err;
    if (stable_cnt >= 3) {
      break;
    }
  }
  return {best_err, best_poles};
}

void OraSolver::numfit(Eigen::Index n) {
  const auto m = getFreqNum();
  mat_Q_.block(0, 0, m, 1) = den_.real();
  mat_Q_.block(m, 0, m, 1) = den_.imag();
  const double m_sqrt = std::sqrt(static_cast<double>(m));
  mat_Q_.col(0) = mat_Q_.col(0).array() * m_sqrt / den_.norm();

  for (Eigen::Index k = 0; k != n; ++k) {
    Eigen::VectorXd q1 = mat_Q_.col(k).head(m);
    Eigen::VectorXd q2 = mat_Q_.col(k).tail(m);
    Eigen::VectorXd q = Eigen::VectorXd(2 * m);
    q.head(m) = -omegas_.array() * q2.array();
    q.tail(m) = omegas_.array() * q1.array();

    if (k > 0) {
      mat_H_(k - 1, k) = -mat_H_(k, k - 1);
      q = q - mat_H_(k - 1, k) * mat_Q_.col(k - 1);
    }
    mat_H_(k + 1, k) = q.norm() / m_sqrt;
    mat_Q_.col(k + 1) = q / mat_H_(k + 1, k);
  }
}

void OraSolver::denfit(Eigen::Index n, Eigen::Index d) {
  const auto m = getFreqNum();
  numfit(n);
  Eigen::MatrixXd mat_Qd1 = mat_Q_.block(0, 0, m, d + 1);
  Eigen::MatrixXd mat_Qd2 = mat_Q_.block(m, 0, m, d + 1);
  Eigen::MatrixXd mat_A = Eigen::MatrixXd(ndata_ * (d + 1), d + 1);

#pragma omp parallel for
  for (Eigen::Index k = 0; k != ndata_; ++k) {
    Eigen::VectorXd fr = getSqmVectorMap(k).real();
    Eigen::VectorXd fi = getSqmVectorMap(k).imag();
    Eigen::MatrixXd mat_Ar = Eigen::MatrixXd(2 * m, d + 1);

    mat_Ar.block(0, 0, m, mat_Ar.cols()) =
        -(fr.array().replicate(1, mat_Qd1.cols()) * mat_Qd1.array() -
          fi.array().replicate(1, mat_Qd2.cols()) * mat_Qd2.array());
    mat_Ar.block(m, 0, m, mat_Ar.cols()) =
        -(fi.array().replicate(1, mat_Qd1.cols()) * mat_Qd1.array() +
          fr.array().replicate(1, mat_Qd2.cols()) * mat_Qd2.array());
    mat_Ar = mat_Ar - mat_Q_ * (mat_Q_.transpose() * mat_Ar) / m;

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(mat_Ar);
    mat_A.block(k * (d + 1), 0, d + 1, d + 1) =
        qr.matrixQR().topRows(d + 1).triangularView<Eigen::Upper>();
  }

  Eigen::BDCSVD<Eigen::MatrixXd> svd(mat_A, Eigen::ComputeThinU |
                                                   Eigen::ComputeThinV);
  Eigen::RowVectorXd row_vector = Eigen::RowVectorXd::Unit(d, 0).reverse();
  const Eigen::VectorXd c = svd.matrixV().col(d);
  poles_ = (mat_H_.block(0, 0, d, d) -
            mat_H_(d, d - 1) * (1 / c(d)) * c.head(d) * row_vector)
               .eigenvalues();
  for (Eigen::Index i = 0; i != poles_.size(); ++i) {
    if (poles_(i).real() > 0) {
      poles_(i) = -poles_(i);
    }
  }
  double mean_abs_s = omegas_.array().abs().mean();

#pragma omp parallel for
  for (Eigen::Index i = 0; i != den_.size(); ++i) {
    den_(i) =
        1.0 /
        ((kImagUnit * omegas_(i) - poles_.array()).array() / mean_abs_s).prod();
  }
}

void OraSolver::calculateModel(const Eigen::VectorXcd &poles) noexcept {

  Eigen::VectorXcd complex_poles;
  Eigen::VectorXd real_poles;

  for (Eigen::Index i = 0; i != poles.size(); ++i) {
    if (poles(i).imag() == 0) {
      real_poles.conservativeResize(real_poles.size() + 1);
      real_poles(real_poles.size() - 1) = poles(i).real();
    } else if (poles(i).imag() > 0) {
      complex_poles.conservativeResize(complex_poles.size() + 1);
      complex_poles(complex_poles.size() - 1) = poles(i);
    }
  }

  Eigen::MatrixXcd phi_real;
  Eigen::MatrixXcd phi_complex;
  calculatePhiMatrices(real_poles, complex_poles, phi_real, phi_complex);

  const auto n_r = real_poles.size();
  const auto n_c = complex_poles.size();
  const auto freq_num = getFreqNum();
  const auto port_num = getPortNum();

  const auto poles_num = real_poles.size() + complex_poles.size() * 2;
  Eigen::MatrixXd mat_phi = Eigen::MatrixXd(2 * getFreqNum(), poles_num + 1);

  mat_phi.col(0).head(freq_num) = Eigen::VectorXd::Ones(freq_num);
  mat_phi.block(0, 1, freq_num, n_r).noalias() = phi_real.real();
  mat_phi.block(0, n_r + 1, freq_num, 2 * n_c).noalias() = phi_complex.real();
  mat_phi.block(freq_num, 1, freq_num, n_r).noalias() = phi_real.imag();
  mat_phi.block(freq_num, n_r + 1, freq_num, 2 * n_c).noalias() =
      phi_complex.imag();

  Eigen::MatrixX<Eigen::VectorXd> c_H(port_num, port_num);
  Eigen::MatrixXd mat_phi_gram = mat_phi.transpose() * mat_phi;
  Eigen::MatrixXd r0 = Eigen::MatrixXd(port_num, port_num);

  auto mat_S_realimag = Eigen::MatrixX<Eigen::VectorXd>(port_num, port_num);
#pragma omp parallel for collapse(2)
  for (Eigen::Index q = 0; q != port_num; ++q) {
    for (Eigen::Index m = 0; m != port_num; ++m) {
      auto vec_Hqm = getSqmVectorMap(q, m);
      Eigen::VectorXd vec_Hqm_real = vec_Hqm.real();
      Eigen::VectorXd vec_Hqm_imag = vec_Hqm.imag();

      Eigen::VectorXd vec_Hqm_realimag(2 * freq_num);
      vec_Hqm_realimag << vec_Hqm_real, vec_Hqm_imag;

      mat_S_realimag(q, m) = vec_Hqm_realimag;
    }
  }
#pragma omp parallel for collapse(2)
  for (Eigen::Index q = 0; q != port_num; q++) {
    for (Eigen::Index m = 0; m != port_num; m++) {
      const auto &blsq = mat_S_realimag(q, m);
      c_H(q, m) = mat_phi_gram.llt().solve(mat_phi.transpose() * blsq);
      r0(q, m) = c_H(q, m)(0);
    }
  }

  solve_res_.model.poles_real = std::move(real_poles);
  solve_res_.model.poles_complex = std::move(complex_poles);
  if (config_.exact_dc && omegas_(0) == 0) {
#pragma omp parallel for collapse(2)
    for (Eigen::Index q = 0; q != port_num; q++) {
      for (Eigen::Index m = 0; m != port_num; m++) {
        r0(q, m) = mat_S_realimag(q, m)(0) -
                   mat_phi.row(0).tail(poles_num) * c_H(q, m).tail(poles_num);
      }
    }
  }

  solve_res_.model.r0 = std::move(r0);
  solve_res_.model.tensor_Rr =
      Eigen::Tensor<double, 3>(n_r, port_num, port_num);
  solve_res_.model.tensor_Rc =
      Eigen::Tensor<std::complex<double>, 3>(n_c, port_num, port_num);

#pragma omp parallel for collapse(2)
  for (Eigen::Index q = 0; q != port_num; ++q) {
    for (Eigen::Index m = 0; m != port_num; ++m) {
      const Eigen::VectorXd &c_Hqm = c_H(q, m);
      for (Eigen::Index i = 0; i != n_r; ++i) {
        solve_res_.model.tensor_Rr(i, q, m) = c_Hqm(i + 1);
      }

      for (Eigen::Index i = 0; i != n_c; ++i) {
        solve_res_.model.tensor_Rc(i, q, m) =
            c_Hqm(n_r + 2 * i + 1) + kImagUnit * c_Hqm(n_r + 2 * i + 2);
      }
    }
  }
}