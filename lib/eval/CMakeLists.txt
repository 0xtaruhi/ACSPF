add_library(EvalLib
    Eval.cpp
)

target_link_libraries(EvalLib PUBLIC Eigen3::Eigen fmt::fmt spdlog::spdlog pthread)
target_include_directories(EvalLib PRIVATE ${CMAKE_SOURCE_DIR}/include)

if (${USE_MKL})
    target_link_libraries(EvalLib
        PUBLIC
        -Wl,--start-group
        ${MKLROOT}/lib/libmkl_intel_lp64.a
        ${MKLROOT}/lib/libmkl_gnu_thread.a
        ${MKLROOT}/lib/libmkl_core.a
        -Wl,--end-group
    )

    target_compile_options(EvalLib PUBLIC -m64 -I${MKLROOT}/include)
    target_link_libraries(EvalLib PUBLIC m dl)
endif()

add_library(VecFit::EvalLib ALIAS EvalLib)
