cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(melotts)

set(CMAKE_CXX_STANDARD 14)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -g -O0")
    add_definitions(-D__AX_SKEL_DEBUG__)
elseif (CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -O2 -fdata-sections -ffunction-sections")
endif()

include(cmake/msp_dependencies.cmake)

# onnxruntime
include_directories(onnxruntime/include/onnxruntime)
include_directories(onnxruntime/include/onnxruntime/core/session)
link_directories(onnxruntime/lib)


include_directories(${MSP_INC_DIR})
link_directories(${MSP_LIB_DIR})

include_directories(src)
aux_source_directory(src SRC)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/bin)
add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp ${SRC})
target_link_libraries(${PROJECT_NAME} ${MSP_LIBS} onnxruntime)

file(COPY onnxruntime/lib/libonnxruntime.so DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
file(COPY onnxruntime/lib/libonnxruntime.so.1.14.0 DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

install(TARGETS ${PROJECT_NAME}
        RUNTIME
            DESTINATION bin)