add_subdirectory(parser)
add_subdirectory(solver)
add_subdirectory(eval)

add_library(VecFitLib INTERFACE)
target_link_libraries(VecFitLib INTERFACE VecFit::ParserLib VecFit::SolverLib VecFit::EvalLib)
target_include_directories(VecFitLib INTERFACE ${CMAKE_SOURCE_DIR}/include)

if(GEN_PYTHON_LIB)
    add_subdirectory(pybind)
endif()