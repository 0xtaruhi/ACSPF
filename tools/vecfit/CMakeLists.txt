add_executable(vecfit vecfit.cpp Config.cpp)

target_link_libraries(vecfit PRIVATE VecFitLib)
target_include_directories(vecfit PRIVATE ${CMAKE_SOURCE_DIR}/thirdparty/argparse/include)

install(TARGETS vecfit
    RUNTIME DESTINATION bin
)
