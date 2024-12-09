#include "Eval.h"

#include <Eigen/Dense>
#include <Eigen/src/SVD/JacobiSVD.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include <fstream>
#include <iomanip>

#include "Model.h"
#include "utils/Logger.h"
#include "utils/dtoa_milo.h"
#include "utils/indicators.hpp"

EvalRes evalFit(const Model &model, const Eigen::VectorXd &omegas,
                const Eigen::Tensor<std::complex<double>, 3> &orig_h_tensor) {
  const auto poles_num = model.getPolesNum();
  Eigen::Tensor<std::complex<double>, 3> fitted_h_tensor =
      model.calHResponse(omegas);

  logger::trace("H response calculated");

  const auto k_bar = orig_h_tensor.dimension(0);
  const auto q_bar = orig_h_tensor.dimension(1);
  const auto m_bar = orig_h_tensor.dimension(2);

  // calculate the rms error
  Eigen::VectorXd diff_norm_vec(k_bar);
  Eigen::VectorXd orig_norm_vec(k_bar);
#pragma omp parallel for
  for (Eigen::Index i = 0; i != k_bar; ++i) {
    Eigen::Tensor<std::complex<double>, 2> diff_mat =
        fitted_h_tensor.chip(i, 0) - orig_h_tensor.chip(i, 0);
    Eigen::Map<Eigen::MatrixXcd> diff_mat_map(diff_mat.data(), q_bar, m_bar);
    Eigen::Tensor<std::complex<double>, 2> orig_mat = orig_h_tensor.chip(i, 0);
    Eigen::Map<Eigen::MatrixXcd> orig_mat_map(orig_mat.data(), q_bar, m_bar);
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd_diff(diff_mat_map);
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd_orig(orig_mat_map);
    diff_norm_vec(i) = svd_diff.singularValues()(0);
    orig_norm_vec(i) = svd_orig.singularValues()(0);
  }

  double err = diff_norm_vec.sum() / orig_norm_vec.sum();
  double kerr = poles_num * err;

  double svd_r0 =
      Eigen::JacobiSVD<Eigen::MatrixXd>(model.r0).singularValues().maxCoeff();

  return {std::move(fitted_h_tensor), err, kerr,
          diff_norm_vec(0) / orig_norm_vec(0), svd_r0};
}

double getErr(const Eigen::Tensor<std::complex<double>, 3> &orig_h_tensor,
              const Eigen::Tensor<std::complex<double>, 3> &fitted_h_tensor) {
  const auto k_bar = orig_h_tensor.dimension(0);
  const auto q_bar = orig_h_tensor.dimension(1);
  const auto m_bar = orig_h_tensor.dimension(2);

  Eigen::VectorXd diff_norm_vec(k_bar);
  Eigen::VectorXd orig_norm_vec(k_bar);

#pragma omp parallel for
  for (Eigen::Index i = 0; i != k_bar; ++i) {
    Eigen::Tensor<std::complex<double>, 2> diff_mat =
        fitted_h_tensor.chip(i, 0) - orig_h_tensor.chip(i, 0);
    Eigen::Map<Eigen::MatrixXcd> diff_mat_map(diff_mat.data(), q_bar, m_bar);
    Eigen::Tensor<std::complex<double>, 2> orig_mat = orig_h_tensor.chip(i, 0);
    Eigen::Map<Eigen::MatrixXcd> orig_mat_map(orig_mat.data(), q_bar, m_bar);
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd_diff(diff_mat_map);
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd_orig(orig_mat_map);
    diff_norm_vec(i) = svd_diff.singularValues()(0);
    orig_norm_vec(i) = svd_orig.singularValues()(0);
  }

  double err = diff_norm_vec.sum() / orig_norm_vec.sum();
  return err;
}

void EvalRes::printToLog() const {
  if (err > 1e-1) {
    logger::warn("[** FAIL **] Error: {:.6f}%, larger than 10%", err * 100);
  } else {
    logger::info("[** PASS **] Error: {:.6f}%", err * 100);
  }

  if (freq0_err > 1e-10) {
    logger::warn("[** FAIL **] Error at f=0: {}, larger than 1e-10", freq0_err);
  } else {
    logger::info("[** PASS **] Error at f=0: {}", freq0_err);
  }

  if (svd_r0 > 1) {
    logger::warn("[** FAIL **] svd of hinf is larger than 1: {}", svd_r0);
  } else {
    logger::info("[** PASS **] svd of hinf: {}", svd_r0);
  }

  logger::info("K: {}", kerr);
}

namespace {
const char *double2String(double value) {
  static char buffer[32];
  dtoa_milo(value, buffer);
  return buffer;
}
} // namespace

void writeRefFile(const Eigen::Tensor<std::complex<double>, 3> &orig_h_tensor,
                  const Eigen::Tensor<std::complex<double>, 3> &fitted_h_tensor,
                  const Eigen::ArrayXd &freqs, std::string_view filename) {
  const auto k_bar = orig_h_tensor.dimension(0);
  const auto q_bar = orig_h_tensor.dimension(1);
  const auto m_bar = orig_h_tensor.dimension(2);

  indicators::ProgressBar bar{
      indicators::option::BarWidth{40},
      indicators::option::Start{"["},
      indicators::option::Fill{"="},
      indicators::option::Lead{">"},
      indicators::option::Remainder{" "},
      indicators::option::End{"]"},
      indicators::option::PostfixText{"Writing Reference File..."},
      indicators::option::ShowElapsedTime{true},
      indicators::option::ShowRemainingTime{true},
  };

  std::fstream file(filename.data(), std::ios::out);

  file << "Freqs,S_row,S_col,Real,Imag,Orig_Real,Orig_Imag,Diff_ratio" << '\n';
  file << std::setprecision(16);

  for (Eigen::Index q = 0; q != q_bar; ++q) {
    for (Eigen::Index m = 0; m != m_bar; ++m) {
      for (Eigen::Index i = 0; i != k_bar; ++i) {
        file << double2String(freqs(i)) << ',' << q << ',' << m << ',';
        file << double2String(fitted_h_tensor(i, q, m).real()) << ','
             << double2String(fitted_h_tensor(i, q, m).imag()) << ',';
        file << double2String(orig_h_tensor(i, q, m).real()) << ','
             << double2String(orig_h_tensor(i, q, m).imag()) << ',';
        double diff_ratio =
            std::abs(fitted_h_tensor(i, q, m) - orig_h_tensor(i, q, m)) /
            std::abs(orig_h_tensor(i, q, m));
        file << double2String(diff_ratio) << '\n';
      }
      bar.set_progress(static_cast<double>(q * m_bar + m + 1) /
                       (q_bar * m_bar) * 100);
    }
  }
}
