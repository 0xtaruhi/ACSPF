#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>

#include <filesystem>
#include <string_view>
#include <utility>

#include <fmt/format.h>

#include "Eval.h"
#include "Model.h"
#include "TouchStoneParser.h"
#include "VFSolver.h"

namespace py = pybind11;

int add(int i, int j) { return i + j; }

auto parse(std::string_view filepath) -> py::object {
  if (!std::filesystem::exists(filepath.data())) {
    throw std::runtime_error(fmt::format("File {} is not exist", filepath));
  }
  TouchStoneParser parser(filepath);
  auto parse_res = parser.parse();

  // Convert OrigInfo to Python object
  py::dict orig_info_dict;
  orig_info_dict["freqs"] = py::cast(parse_res.freqs);
  orig_info_dict["sparams"] = py::cast(parse_res.s_params);

  return orig_info_dict;
}

struct Network {
  Network(std::string_view filepath) {
    if (!std::filesystem::exists(filepath.data())) {
      throw std::runtime_error(fmt::format("File {} is not exist", filepath));
    }
    TouchStoneParser parser(filepath.data());
    auto parse_res = parser.parse();
    freqs = std::move(parse_res.freqs);
    sparams = std::move(parse_res.s_params);
    omegas = freqs * 2 * M_PI;
  }

  Model fit(int num_poles, int max_iters) {
    VFSolver solver(freqs, sparams, {max_iters, num_poles, true, true});
    auto res = solver.solve();
    return res.model;
  }

  Eigen::VectorXd freqs;
  Eigen::VectorXd omegas;
  Eigen::Tensor<std::complex<double>, 3> sparams;
};

PYBIND11_MODULE(vecfitpy, m) {
  // Add class
  py::class_<Network>(m, "Network")
      .def(py::init<std::string_view>(), py::arg("filepath"))
      .def_readwrite("freqs", &Network::freqs)
      .def_readwrite("sparams", &Network::sparams)
      .def("fit", &Network::fit, py::arg("poles") = 10,
           py::arg("max_iters") = 100);

  py::class_<Model>(m, "Model")
      .def(py::init<>())
      .def(py::init<std::string_view>(), py::arg("filepath"))
      .def("write", &Model::writeToFile, py::arg("filepath"))
      .def_readwrite("poles_real", &Model::poles_real)
      .def_readwrite("poles_complex", &Model::poles_complex)
      .def_readwrite("r0", &Model::r0)
      .def_readwrite("residue_real", &Model::tensor_Rr)
      .def_readwrite("residue_complex", &Model::tensor_Rc)
      .def("cal_response",
           [](Model &self, const Eigen::VectorXd &freqs) {
             return self.calHResponse(freqs * 2 * M_PI);
           })
      .def("eval", [](Model &self, const Network &network) {
        auto fitted_h_tensor = self.calHResponse(network.omegas);
        auto eval_res = evalFit(self, network.omegas, network.sparams);
        py::dict eval_dict;
        eval_dict["err"] = eval_res.err;
        eval_dict["kerr"] = eval_res.kerr;
        eval_dict["freq0_err"] = eval_res.freq0_err;
        eval_dict["svd_r0"] = eval_res.svd_r0;
        return eval_dict;
      });
}
