cmake_policy(SET CMP0148 OLD)
find_package(pybind11)

pybind11_add_module(vecfitpy vecfitpy.cpp)

target_link_libraries(vecfitpy PRIVATE VecFitLib)

install(TARGETS vecfitpy DESTINATION python)
