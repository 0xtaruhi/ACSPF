add_library(ParserLib
    FileMemoryMap.cpp
    TouchStoneParser.cpp
    Model.cpp
)

target_link_libraries(ParserLib PUBLIC Eigen3::Eigen fmt::fmt spdlog::spdlog pthread)
target_include_directories(ParserLib PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(VecFit::ParserLib ALIAS ParserLib)
