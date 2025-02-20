cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(melotts)
option(BUILD_ONNX "Build on onnx runtime." OFF)
option(BUILD_NNCASE "Build on nncase." ON)

if (BUILD_ONNX)
    add_definitions(-DONNX)
endif()

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -g -O0")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -O3 -fdata-sections -ffunction-sections")
endif()

set(arch "x86_64")
if (CMAKE_CROSSCOMPILING)
    set(arch "riscv64")
endif()

if (BUILD_ONNX)
    include_directories(3rd_party/onnxruntime/${arch}/include/)
    link_directories(3rd_party/onnxruntime/${arch}/lib)
else()
    if(CMAKE_CROSSCOMPILING)
        set(NNCASE_PATH ${CMAKE_SOURCE_DIR}/3rd_party/nncase/riscv64)
        link_directories(${CMAKE_SOURCE_DIR}/3rd_party/mmz/riscv64)
    else()
        set(NNCASE_PATH ${CMAKE_SOURCE_DIR}/3rd_party/nncase/x86_64)
    endif()
    include_directories(3rd_party
                    ${NNCASE_PATH}/include
                    ${NNCASE_PATH}/include/nncase/runtime
                    )
    link_directories(${NNCASE_PATH}/lib)
endif()

include_directories(src)
if (BUILD_ONNX)
    set(SRC src/OnnxWrapper.cpp)
endif()

set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/bin)
add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp ${SRC})

if (BUILD_ONNX)
    target_link_libraries(${PROJECT_NAME} onnxruntime onnxruntime_providers_shared)
    file(GLOB onnx_lib_files  "3rd_party/onnxruntime/${arch}/lib/*")
    file(COPY ${onnx_lib_files} DESTINATION ${CMAKE_INSTALL_PREFIX})
else()
    if(CMAKE_CROSSCOMPILING)
        target_link_libraries(${PROJECT_NAME} nncase.rt_modules.k230 Nncase.Runtime.Native functional_k230 mmz)
    else()
        target_link_libraries(${PROJECT_NAME} Nncase.Runtime.Native)
        file(GLOB nncase_lib_files  "NNCASE_PATH/lib/*")
        file(COPY ${nncase_lib_files} DESTINATION ${CMAKE_INSTALL_PREFIX})
    endif()
endif()

install(TARGETS ${PROJECT_NAME}
        RUNTIME
            DESTINATION ./)
set_target_properties(${PROJECT_NAME}
    PROPERTIES
    INSTALL_RPATH "$ORIGIN/"
)