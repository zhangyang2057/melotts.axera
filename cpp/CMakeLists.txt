cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(melotts)

set(CMAKE_CXX_STANDARD 14)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -g -O0")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -O3 -fdata-sections -ffunction-sections")
endif()

# onnxruntime
include_directories(onnxruntime/include/)
include_directories(onnxruntime/include/onnxruntime/core/session)
link_directories(onnxruntime/lib)


include_directories(src)
aux_source_directory(src SRC)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/bin)
add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp ${SRC})
target_link_libraries(${PROJECT_NAME} onnxruntime onnxruntime_providers_shared)

file(COPY onnxruntime/lib/libonnxruntime.so DESTINATION ${CMAKE_INSTALL_PREFIX})
file(COPY onnxruntime/lib/libonnxruntime.so.1 DESTINATION ${CMAKE_INSTALL_PREFIX})
file(COPY onnxruntime/lib/libonnxruntime.so.1.19.2 DESTINATION ${CMAKE_INSTALL_PREFIX})
file(COPY onnxruntime/lib/libonnxruntime_providers_shared.so DESTINATION ${CMAKE_INSTALL_PREFIX})

install(TARGETS ${PROJECT_NAME}
        RUNTIME
            DESTINATION ./)
set_target_properties(${PROJECT_NAME}
    PROPERTIES
    INSTALL_RPATH "$ORIGIN/"
)