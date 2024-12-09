add_library(SolverLib
    Solver.cpp
    VFSolver.cpp
    OraSolver.cpp
)

target_include_directories(SolverLib PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(SolverLib PUBLIC Eigen3::Eigen fmt::fmt spdlog::spdlog pthread)
target_include_directories(SolverLib PRIVATE ${CMAKE_SOURCE_DIR}/thirdparty/argparse/include)

if(${USE_MKL})
    target_link_libraries(SolverLib
        PUBLIC
        -Wl,--start-group
        ${MKLROOT}/lib/libmkl_intel_lp64.a
        ${MKLROOT}/lib/libmkl_gnu_thread.a
        ${MKLROOT}/lib/libmkl_core.a
        -Wl,--end-group
    )

    target_compile_options(SolverLib PUBLIC -m64 -I${MKLROOT}/include)
    target_link_libraries(SolverLib PUBLIC m dl)
endif()

if(USE_JEMALLOC)
    target_link_directories(SolverLib PUBLIC ${JEMALLOC_ROOT}/lib)
    target_include_directories(SolverLib PUBLIC ${JEMALLOC_ROOT}/include)
    target_link_libraries(SolverLib PUBLIC jemalloc)
endif()

add_library(VecFit::SolverLib ALIAS SolverLib)