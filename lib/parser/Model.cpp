#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include <Eigen/Core>

#include "Model.h"
#include "utils/Logger.h"
#include "utils/Parse.h"
#include "utils/dtoa_milo.h"

namespace {
const char *double2String(double value) {
  static char buffer[32];
  dtoa_milo(value, buffer);
  return buffer;
}
} // namespace

Model::Model(std::string_view filename) {
  // Load model from file
  std::string line;
  std::ifstream file(filename.data());

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    // Parse line
    if (line.find("Poles") != std::string::npos) {
      int n_poles;
      int n_c, n_r;
      // from line
      std::string poles;
      iss >> poles >> n_poles >> n_c >> n_r;

      assert(n_poles == n_c * 2 + n_r && "Invalid number of poles");
      this->poles_complex.resize(n_c);
      this->poles_real.resize(n_r);

      for (int i = 0; i < n_c; i++) {
        std::getline(file, line);
        std::istringstream iss(line);
        double real, imag;
        iss >> real >> imag;
        this->poles_complex[i] = std::complex<double>(real, imag);
        // ignore next line
        std::getline(file, line);
      }

      for (int i = 0; i < n_r; i++) {
        std::getline(file, line);
        std::istringstream iss(line);
        double real;
        iss >> real;
        this->poles_real[i] = real;
      }
    } else if (line.find("Residues") != std::string::npos) {
      int n_ports;
      std::string residues;
      iss >> residues >> n_ports;

      const auto n_c = this->poles_complex.size();
      const auto n_r = this->poles_real.size();

      this->tensor_Rr =
          Eigen::Tensor<double, 3>(n_r, n_ports, n_ports);
      this->tensor_Rc =
          Eigen::Tensor<std::complex<double>, 3>(n_c, n_ports, n_ports);
      for (int i = 0; i < n_c; i++) {
        std::getline(file, line);
        const char *data_ptr = line.data();
        utils::skipIfIsSpace(data_ptr);
        for (int j = 0; j < n_ports; j++) {
          for (int k = 0; k < n_ports; k++) {
            double real, imag;
            real = utils::parseDouble(data_ptr);
            utils::skipIfIsSpace(data_ptr);
            imag = utils::parseDouble(data_ptr);
            utils::skipIfIsSpace(data_ptr);
            this->tensor_Rc(i, j, k) = std::complex<double>(real, imag);
          }
        }
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      }

      for (int i = 0; i < n_r; i++) {
        std::getline(file, line);
        const char *data_ptr = line.data();
        utils::skipIfIsSpace(data_ptr);
        for (int j = 0; j < n_ports; j++) {
          for (int k = 0; k < n_ports; k++) {
            double real;
            real = utils::parseDouble(data_ptr);
            utils::skipIfIsSpace(data_ptr);
            this->tensor_Rr(i, j, k) = real;
          }
        }
      }

    } else if (line.find("Hinf") != std::string::npos) {
      std::string hinf;
      int n_ports;
      iss >> hinf >> n_ports;
      std::getline(file, line);
      const char *data_ptr = line.data();
      this->r0 = Eigen::MatrixXd(n_ports, n_ports);
      for (int i = 0; i < n_ports; i++) {
        for (int j = 0; j < n_ports; j++) {
          double real;
          utils::skipIfIsSpace(data_ptr);
          real = utils::parseDouble(data_ptr);
          this->r0(i, j) = real;
        }
      }
    }
  }
}

void Model::writeToFile(std::string_view filename) {
  std::ofstream file(filename.data());
  if (!file.is_open()) {
    logger::error("Failed to open file {} for writing.", filename);
    return;
  }

  // Write poles
  const auto complex_poles_num_half = poles_complex.size();
  const auto real_poles_num = poles_real.size();
  const auto poles_num = complex_poles_num_half * 2 + real_poles_num;
  file << "Poles: " << poles_num << ' ' << complex_poles_num_half << ' '
       << real_poles_num << '\n';

  for (Eigen::Index i = 0; i != complex_poles_num_half; ++i) {
    file << double2String(poles_complex(i).real()) << '\t'
         << double2String(poles_complex(i).imag()) << '\n';
    file << double2String(poles_complex(i).real()) << '\t'
         << double2String(-poles_complex(i).imag()) << '\n';
  }

  for (Eigen::Index i = 0; i != real_poles_num; ++i) {
    file << double2String(poles_real(i)) << '\n';
  }

  // Write Residues
  file << "Residues: " << tensor_Rc.dimension(1) << '\n';

  for (Eigen::Index i = 0; i != complex_poles_num_half; ++i) {
    const Eigen::Tensor<std::complex<double>, 2> &chipped_tensor_Rc =
        tensor_Rc.chip(i, 0);
    for (Eigen::Index q = 0; q != tensor_Rc.dimension(1); ++q) {
      for (Eigen::Index m = 0; m != tensor_Rc.dimension(2); ++m) {
        file << double2String(chipped_tensor_Rc(q, m).real()) << '\t'
             << double2String(chipped_tensor_Rc(q, m).imag()) << '\t';
      }
    }

    file << '\n';

    for (Eigen::Index q = 0; q != tensor_Rc.dimension(1); ++q) {
      for (Eigen::Index m = 0; m != tensor_Rc.dimension(2); ++m) {
        file << double2String(chipped_tensor_Rc(q, m).real()) << '\t'
             << double2String(-chipped_tensor_Rc(q, m).imag()) << '\t';
      }
    }

    file << '\n';
  }

  for (Eigen::Index i = 0; i != real_poles_num; ++i) {
    const Eigen::Tensor<double, 2> &chipped_tensor_Rr = tensor_Rr.chip(i, 0);
    for (Eigen::Index q = 0; q != tensor_Rr.dimension(1); ++q) {
      for (Eigen::Index m = 0; m != tensor_Rr.dimension(2); ++m) {
        file << double2String(chipped_tensor_Rr(q, m)) << '\t';
      }
    }
    file << '\n';
  }

  file << "Hinf: " << r0.rows() << '\n';

  for (Eigen::Index q = 0; q != r0.rows(); ++q) {
    for (Eigen::Index m = 0; m != r0.cols(); ++m) {
      file << double2String(r0(q, m)) << '\t';
    }
  }

  file << '\n';

  file.close();
}
