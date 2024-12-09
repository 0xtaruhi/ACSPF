add_executable(evaltool
    evaltool.cpp
    Config.cpp
)

target_link_libraries(evaltool PRIVATE VecFit::ParserLib VecFit::EvalLib)
target_include_directories(evaltool PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(evaltool PRIVATE ${CMAKE_SOURCE_DIR}/thirdparty/argparse/include)

install(TARGETS evaltool
    RUNTIME DESTINATION bin
)
