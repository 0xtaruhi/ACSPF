#include <cmath>

#include <Eigen/Dense>
#include <limits>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Model.h"
#include "VFSolver.h"
#include "utils/Logger.h"

static constexpr double kAlpha = 0.01;
static constexpr double kEps = std::numeric_limits<double>().epsilon();

namespace {

using EigenIndex = decltype(Eigen::Index());

/**
 * @brief Generate initial poles for the vector fitting algorithm.
 * @param start_freq Start frequency.
 * @param end_freq End frequency.
 * @param half_poles_num Half of the number of poles.
 * @return Array of complex poles.
 * @details The poles are generated in the following way:
 * poles[n] = (-alpha, 1) * (fS + (fE - fS) / (N / 2 - 1) * n), n = 0, 1, ...,
 * N / 2 - 1, poles[n] = conj(poles[n - N / 2]), n = N / 2, ..., N - 1, where N
 * is the number of poles, fS is the start frequency, fE is the end frequency.
 */
Eigen::VectorXcd generateInitComplexPoles(double min_omega, double max_omega,
                                          int half_poles_num) {
  Eigen::VectorXcd poles(half_poles_num);

  constexpr auto basic_complex = std::complex<double>(-kAlpha, 1.0);
  double diff = max_omega - min_omega;

  if (min_omega == 0) {
    for (EigenIndex i = 0; i != half_poles_num; ++i) {
      poles(i) = basic_complex * max_omega * static_cast<double>(i + 1) /
                 static_cast<double>(half_poles_num);
    }
  } else {
    for (EigenIndex i = 0; i != half_poles_num; ++i) {
      poles(i) = basic_complex * (min_omega + diff / (half_poles_num - 1) * i);
    }
  }

  return poles;
}

Eigen::VectorXd generateInitRealPoles(double max_omega) {
  Eigen::VectorXd poles(1);
  poles(0) = -kAlpha * max_omega;
  return poles;
}

static constexpr std::complex<double> kImagUnit(0.0, 1.0);
} // namespace

VFSolver::VFSolver(const Eigen::VectorXd &freqs,
                   const Eigen::Tensor<std::complex<double>, 3> &h_tensor,
                   SolverConfig config)
    : Solver(freqs, h_tensor, std::move(config)) {
  const auto freq_num = getFreqNum();
  const auto port_num = getPortNum();
  mat_Sqm_realimag_ = Eigen::MatrixX<Eigen::VectorXd>(port_num, port_num);

#pragma omp parallel for collapse(2)
  for (EigenIndex q = 0; q != port_num; ++q) {
    for (EigenIndex m = 0; m != port_num; ++m) {
      auto vec_Hqm = getSqmVectorMap(q, m);

      Eigen::VectorXd vec_Hqm_real = vec_Hqm.real();
      Eigen::VectorXd vec_Hqm_imag = vec_Hqm.imag();

      Eigen::VectorXd vec_Hqm_realimag(2 * freq_num);
      vec_Hqm_realimag << vec_Hqm_real, vec_Hqm_imag;

      mat_Sqm_realimag_(q, m) = vec_Hqm_realimag;
    }
  }
}

SolveRes &VFSolver::solve() noexcept {
  logger::info("Solving...");

  if (config_.reduced_columns) {
    ndata_ = reduceColumns();
    logger::trace("Reduced columns number: {}", ndata_);
  } else {
    ndata_ = getPortNum() * getPortNum();
  }

  auto pole_num = getPoleNum();
  solveWithFixedPolesNum(pole_num);
  return solve_res_;
}

void VFSolver::solveWithFixedPolesNum(int poles_num) noexcept {
  logger::trace("Solving with {} poles...", poles_num);
  const auto min_omega = omegas_(0);
  const auto max_omega = omegas_(omegas_.size() - 1);

  auto complex_poles =
      generateInitComplexPoles(min_omega, max_omega, poles_num / 2);

  auto real_poles = (poles_num % 2 == 1) ? generateInitRealPoles(max_omega)
                                         : Eigen::VectorXd{};

  const auto freq_num = getFreqNum();
  const auto port_num = getPortNum();
  auto n_bar = poles_num; // number of poles

  auto n_r = real_poles.size();    // number of real poles
  auto n_c = complex_poles.size(); // number of complex poles

  // Pre-allocate memory
  Eigen::MatrixX<Eigen::VectorXd> mat_Sqm_realimag(port_num, port_num);
  Eigen::MatrixXd mat_M = Eigen::MatrixXd::Zero(2 * freq_num, 2 * n_bar);
  Eigen::MatrixXd mat_A_lsq = Eigen::MatrixXd::Zero(n_bar * ndata_, n_bar);
  auto vec_b_lsq = Eigen::VectorXd(n_bar * ndata_);

  int iter_num = 0;

  updatePhiMatrices(real_poles, complex_poles);

  // Begin iteration
  while (iter_num != config_.max_iters) {
    iter_num++;
    mat_M.col(0) = Eigen::VectorXd::Zero(2 * freq_num);
    mat_M.block(0, 0, freq_num, n_r).noalias() = phi_real_.real();
    mat_M.block(0, n_r, freq_num, n_c * 2).noalias() = phi_complex_.real();
    mat_M.block(freq_num, 0, freq_num, n_r).noalias() = phi_real_.imag();
    mat_M.block(freq_num, n_r, freq_num, n_c * 2).noalias() =
        phi_complex_.imag();

#pragma omp parallel for firstprivate(mat_M)
    for (Eigen::Index n = 0; n != ndata_; ++n) {
      const auto &vec_Hqm = getSqmVectorMap(n);

      Eigen::MatrixXcd temp_mat_1(freq_num, n_r);
      Eigen::MatrixXcd temp_mat_2(freq_num, n_c * 2);
      for (EigenIndex i = 0; i != freq_num; ++i) {
        temp_mat_1.row(i).noalias() = phi_real_.row(i) * vec_Hqm(i);
        temp_mat_2.row(i).noalias() = phi_complex_.row(i) * vec_Hqm(i);
      }

      mat_M.block(0, n_bar, freq_num, n_r).noalias() = -temp_mat_1.real();
      mat_M.block(0, n_bar + n_r, freq_num, n_c * 2).noalias() =
          -temp_mat_2.real();
      mat_M.block(freq_num, n_bar, freq_num, n_r).noalias() =
          -temp_mat_1.imag();
      mat_M.block(freq_num, n_bar + n_r, freq_num, n_c * 2).noalias() =
          -temp_mat_2.imag();

      Eigen::HouseholderQR<Eigen::MatrixXd> qr(mat_M);
      Eigen::MatrixXd mat_Qqm =
          qr.householderQ() *
          Eigen::MatrixXd::Identity(2 * freq_num, 2 * n_bar);
      const Eigen::TriangularView<const Eigen::MatrixXd, Eigen::Upper> mat_Rqm =
          qr.matrixQR().triangularView<Eigen::Upper>();

      const EigenIndex irow = n_bar * n;
      for (EigenIndex i = 0; i != n_bar; ++i) {
        for (EigenIndex j = i; j != n_bar; ++j) {
          mat_A_lsq(irow + i, j) = mat_Rqm(n_bar + i, n_bar + j);
        }
      }
      vec_b_lsq.segment(irow, n_bar).noalias() =
          mat_Qqm.block(0, n_bar, freq_num, n_bar).transpose() *
              vec_Hqm.real() +
          mat_Qqm.block(freq_num, n_bar, freq_num, n_bar).transpose() *
              vec_Hqm.imag();
    }

    Eigen::VectorXd c_w = mat_A_lsq.householderQr().solve(vec_b_lsq);
    Eigen::VectorXcd w = phi_real_ * c_w.head(n_r) +
                         phi_complex_ * c_w.tail(n_c * 2) +
                         Eigen::VectorXd::Ones(freq_num);

    Eigen::MatrixXd mat_A = Eigen::MatrixXd::Zero(n_bar, n_bar);
    Eigen::VectorXd vec_bw = Eigen::VectorXd::Ones(n_bar);

    for (EigenIndex i = 0; i != n_r; ++i) {
      mat_A(i, i) = real_poles(i);
    }

    for (EigenIndex i = 0; i != n_c; ++i) {
      const int idx = n_r + 2 * i;

      mat_A(idx, idx) = complex_poles(i).real();
      mat_A(idx, idx + 1) = complex_poles(i).imag();
      mat_A(idx + 1, idx) = -complex_poles(i).imag();
      mat_A(idx + 1, idx + 1) = complex_poles(i).real();

      vec_bw(idx) = 2;
      vec_bw(idx + 1) = 0;
    }

    auto poles_new = (mat_A - vec_bw * c_w.transpose()).eigenvalues();
    real_poles.resize(0);
    complex_poles.resize(0);

    EigenIndex real_poles_num = 0;
    EigenIndex complex_poles_num = 0;

    for (const auto &val : poles_new) {
      const auto val_abs = std::abs(val);
      if (std::abs(val.imag()) < kEps * val_abs * 10) {
        real_poles.conservativeResize(real_poles_num + 1);
        real_poles(real_poles_num) = -val_abs;
        ++real_poles_num;
      } else if (val.imag() >= kEps * val_abs * 10) {
        complex_poles.conservativeResize(complex_poles_num + 1);
        complex_poles(complex_poles_num) =
            -std::abs(val.real()) + kImagUnit * val.imag();
        ++complex_poles_num;
      }
    }

    n_r = real_poles.size();
    n_c = complex_poles.size();

    auto poles_err = 1 / std::sqrt(freq_num) *
                     (w.cwiseAbs() - Eigen::VectorXd::Ones(freq_num)).norm();

    updatePhiMatrices(real_poles, complex_poles);

    if (poles_err >= 1e-3 && !(iter_num >= config_.max_iters)) {
      logger::trace("Iteration {} finished due to poles_err = {}, which is "
                    "larger than the threshold {}",
                    iter_num, poles_err, 1e-3);
      continue;
    }

    Eigen::MatrixXd mat_phi = Eigen::MatrixXd(2 * freq_num, n_bar);
    mat_phi.block(0, 0, freq_num, n_r).noalias() = phi_real_.real();
    mat_phi.block(0, n_r, freq_num, 2 * n_c).noalias() = phi_complex_.real();
    mat_phi.block(freq_num, 0, freq_num, n_r).noalias() = phi_real_.imag();
    mat_phi.block(freq_num, n_r, freq_num, 2 * n_c).noalias() =
        phi_complex_.imag();

    Eigen::MatrixX<Eigen::VectorXd> c_H(port_num, port_num);
    Eigen::MatrixXd mat_phi_gram = mat_phi.transpose() * mat_phi;
    Eigen::MatrixXd r0 = Eigen::MatrixXd(port_num, port_num);

#pragma omp parallel for collapse(2)
    for (Eigen::Index q = 0; q != port_num; q++) {
      for (Eigen::Index m = 0; m != port_num; m++) {
        const auto &blsq = mat_Sqm_realimag_(q, m);
        c_H(q, m) = mat_phi_gram.llt().solve(mat_phi.transpose() * blsq);
      }
    }

    if (config_.exact_dc && omegas_(0) == 0) {
#pragma omp parallel for collapse(2)
      for (Eigen::Index q = 0; q != port_num; q++) {
        for (Eigen::Index m = 0; m != port_num; m++) {
          r0(q, m) = mat_Sqm_realimag_(q, m)(0) - mat_phi.row(0) * c_H(q, m);
        }
      }
    } else {
      r0 = Eigen::MatrixXd::Zero(port_num, port_num);
    }

    if (iter_num >= config_.max_iters) {
      solve_res_.model.poles_real = std::move(real_poles);
      solve_res_.model.poles_complex = std::move(complex_poles);

      solve_res_.model.r0 = std::move(r0);
      solve_res_.model.tensor_Rr =
          Eigen::Tensor<double, 3>(n_r, port_num, port_num);
      solve_res_.model.tensor_Rc =
          Eigen::Tensor<std::complex<double>, 3>(n_c, port_num, port_num);

#pragma omp parallel for collapse(2)
      for (Eigen::Index q = 0; q != port_num; ++q) {
        for (EigenIndex m = 0; m != port_num; ++m) {
          const Eigen::VectorXd &c_Hqm = c_H(q, m);
          for (EigenIndex i = 0; i != n_r; ++i) {
            solve_res_.model.tensor_Rr(i, q, m) = c_Hqm(i);
          }

          for (EigenIndex i = 0; i != n_c; ++i) {
            solve_res_.model.tensor_Rc(i, q, m) =
                c_Hqm(n_r + 2 * i) + kImagUnit * c_Hqm(n_r + 2 * i + 1);
          }
        }
      }
      logger::info("Iteration finished.");
      return;
    }

    logger::trace("Iteration {} finished", iter_num);
  }
}

void VFSolver::updatePhiMatrices(
    const Eigen::VectorXd &real_poles,
    const Eigen::VectorXcd &complex_poles) noexcept {
  calculatePhiMatrices(real_poles, complex_poles, phi_real_, phi_complex_);
}
