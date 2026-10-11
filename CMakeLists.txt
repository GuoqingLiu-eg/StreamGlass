cmake_minimum_required(VERSION 3.22)

project(streamglass_runtime VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(STREAMGLASS_ENABLE_TENSORRT "Build the Jetson TensorRT runtime" ON)
option(STREAMGLASS_BUILD_TESTS "Build native unit tests" ON)
option(
    STREAMGLASS_BUILD_SILERO_VAD_EXPERIMENT
    "Build optional Silero VAD probe and shadow tools"
    OFF
)

add_library(streamglass_core INTERFACE)
target_include_directories(
    streamglass_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set(
    STREAMGLASS_KNF_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/seamless_communication/ggml/examples/kaldi-native-fbank"
)

add_library(
    streamglass_knf STATIC
    ${STREAMGLASS_KNF_ROOT}/csrc/feature-fbank.cc
    ${STREAMGLASS_KNF_ROOT}/csrc/feature-functions.cc
    ${STREAMGLASS_KNF_ROOT}/csrc/feature-window.cc
    ${STREAMGLASS_KNF_ROOT}/csrc/fftsg.c
    ${STREAMGLASS_KNF_ROOT}/csrc/mel-computations.cc
    ${STREAMGLASS_KNF_ROOT}/csrc/online-feature.cc
    ${STREAMGLASS_KNF_ROOT}/csrc/rfft.cc
)
target_include_directories(
    streamglass_knf PUBLIC "${STREAMGLASS_KNF_ROOT}/.."
)
target_compile_options(
    streamglass_knf
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Wpedantic>
)

add_library(
    streamglass_audio STATIC
    src/online_fbank.cpp
    src/playback_queue.cpp
    src/utterance_segmenter.cpp
)
target_link_libraries(streamglass_audio PUBLIC streamglass_core streamglass_knf)
target_compile_options(
    streamglass_audio
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
)

add_executable(streamglass-fbank-replay app/fbank_replay.cpp)
target_include_directories(
    streamglass-fbank-replay
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/seamless_communication/ggml/examples"
)
target_link_libraries(streamglass-fbank-replay PRIVATE streamglass_audio)
target_compile_options(
    streamglass-fbank-replay
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
)
install(
    TARGETS streamglass-fbank-replay
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

add_library(streamglass_policy STATIC src/tokenizer_assets.cpp)
target_link_libraries(streamglass_policy PUBLIC streamglass_core)
target_compile_options(
    streamglass_policy
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
)

add_executable(streamglass-tokenizer-inspect app/tokenizer_inspect.cpp)
target_link_libraries(streamglass-tokenizer-inspect PRIVATE streamglass_policy)
target_compile_options(
    streamglass-tokenizer-inspect
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
)
install(
    TARGETS streamglass-tokenizer-inspect
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(STREAMGLASS_ENABLE_TENSORRT)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GIO REQUIRED IMPORTED_TARGET gio-2.0)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_path(
        TENSORRT_INCLUDE_DIR
        NvInfer.h
        HINTS /usr/include/aarch64-linux-gnu /usr/include
        REQUIRED
    )
    find_library(TENSORRT_LIBRARY nvinfer REQUIRED)
    find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin REQUIRED)

    add_library(streamglass_trt STATIC src/trt_engine.cpp)
    target_include_directories(streamglass_trt PUBLIC ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(
        streamglass_trt
        PUBLIC
            streamglass_core
            CUDA::cudart
            ${TENSORRT_LIBRARY}
            ${TENSORRT_PLUGIN_LIBRARY}
    )
    target_compile_options(
        streamglass_trt
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
    )

    add_library(streamglass_cuda STATIC src/cuda_kernels.cu)
    target_link_libraries(streamglass_cuda PUBLIC streamglass_core CUDA::cudart)
    set_target_properties(
        streamglass_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
    )
    target_compile_options(
        streamglass_cuda
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )

    if(STREAMGLASS_BUILD_TESTS)
        add_executable(
            streamglass-t2u-policy-tests
            tests/native/test_t2u_policy.cpp
        )
        target_link_libraries(
            streamglass-t2u-policy-tests PRIVATE streamglass_cuda
        )
        target_compile_options(
            streamglass-t2u-policy-tests
            PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
        )
        add_test(
            NAME streamglass-t2u-policy-tests
            COMMAND streamglass-t2u-policy-tests
        )
    endif()

    add_library(
        streamglass_inference STATIC
        src/decoder_runtime.cpp
        src/encoder_runtime.cpp
        src/streaming_pipeline.cpp
        src/streaming_text_policy.cpp
        src/t2u_runtime.cpp
        src/vocoder_runtime.cpp
    )
    target_link_libraries(
        streamglass_inference
        PUBLIC
            streamglass_trt
            streamglass_cuda
            streamglass_audio
            streamglass_policy
    )
    target_compile_options(
        streamglass_inference
        PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )

    add_executable(streamglass-engine-inspect app/engine_inspect.cpp)
    target_link_libraries(streamglass-engine-inspect PRIVATE streamglass_trt)

    add_executable(streamglass-encoder-replay app/encoder_replay.cpp)
    target_include_directories(
        streamglass-encoder-replay
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/seamless_communication/ggml/examples"
    )
    target_link_libraries(
        streamglass-encoder-replay PRIVATE streamglass_inference
    )
    target_compile_options(
        streamglass-encoder-replay
        PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )

    add_executable(streamglass-decoder-replay app/decoder_replay.cpp)
    target_link_libraries(
        streamglass-decoder-replay PRIVATE streamglass_inference
    )
    target_compile_options(
        streamglass-decoder-replay
        PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )

    add_executable(streamglass-t2u-replay app/t2u_replay.cpp)
    target_link_libraries(streamglass-t2u-replay PRIVATE streamglass_inference)
    target_compile_options(
        streamglass-t2u-replay
        PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )

    add_executable(streamglass-vocoder-replay app/vocoder_replay.cpp)
    target_link_libraries(
        streamglass-vocoder-replay PRIVATE streamglass_inference
    )
    target_compile_options(
        streamglass-vocoder-replay
        PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )

    add_executable(
        streamglass-native-file-replay app/native_file_replay.cpp
    )
    target_include_directories(
        streamglass-native-file-replay
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/seamless_communication/ggml/examples"
    )
    target_link_libraries(
        streamglass-native-file-replay PRIVATE streamglass_inference
    )
    target_compile_options(
        streamglass-native-file-replay
        PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )

    add_library(streamglass_silero_vad STATIC src/silero_vad_trt.cpp)
    target_link_libraries(streamglass_silero_vad PUBLIC streamglass_trt)
    target_compile_options(
        streamglass_silero_vad
        PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )

    add_executable(
        streamglass-native-realtime
        app/native_realtime.cpp
        src/bluez_media_controller.cpp
        src/hfp_button_monitor.cpp
        src/runtime_control.cpp
    )
    target_link_libraries(
        streamglass-native-realtime
        PRIVATE
            streamglass_inference
            streamglass_silero_vad
            PkgConfig::GIO
    )
    target_compile_definitions(
        streamglass-native-realtime
        PRIVATE STREAMGLASS_HAS_SILERO_VAD=1
    )
    target_compile_options(
        streamglass-native-realtime
        PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )

    if(STREAMGLASS_BUILD_SILERO_VAD_EXPERIMENT)
        add_executable(streamglass-silero-vad-probe app/silero_vad_probe.cpp)
        target_link_libraries(
            streamglass-silero-vad-probe
            PRIVATE streamglass_silero_vad
        )
        target_compile_options(
            streamglass-silero-vad-probe
            PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
        )

        add_executable(streamglass-silero-vad-shadow app/silero_vad_shadow.cpp)
        target_link_libraries(
            streamglass-silero-vad-shadow
            PRIVATE streamglass_silero_vad streamglass_audio
        )
        target_compile_options(
            streamglass-silero-vad-shadow
            PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
        )
    endif()

    add_executable(
        streamglass-native-threaded-file app/native_threaded_file.cpp
    )
    target_include_directories(
        streamglass-native-threaded-file
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/seamless_communication/ggml/examples"
    )
    target_link_libraries(
        streamglass-native-threaded-file PRIVATE streamglass_inference
    )
    target_compile_options(
        streamglass-native-threaded-file
        PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )

    install(
        TARGETS
            streamglass-engine-inspect
            streamglass-encoder-replay
            streamglass-decoder-replay
            streamglass-t2u-replay
            streamglass-vocoder-replay
            streamglass-native-file-replay
            streamglass-native-threaded-file
            streamglass-native-realtime
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    if(STREAMGLASS_BUILD_SILERO_VAD_EXPERIMENT)
        install(
            TARGETS
                streamglass-silero-vad-probe
                streamglass-silero-vad-shadow
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/experiments
        )
    endif()
endif()

if(STREAMGLASS_BUILD_TESTS)
    enable_testing()
    add_executable(streamglass-core-tests tests/native/test_core.cpp)
    target_link_libraries(
        streamglass-core-tests PRIVATE streamglass_core streamglass_policy
    )
    target_compile_options(
        streamglass-core-tests
        PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
    add_test(NAME streamglass-core-tests COMMAND streamglass-core-tests)

    add_executable(
        streamglass-audio-tests tests/native/test_audio_frontend.cpp
    )
    target_link_libraries(streamglass-audio-tests PRIVATE streamglass_audio)
    target_compile_options(
        streamglass-audio-tests
        PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
    add_test(NAME streamglass-audio-tests COMMAND streamglass-audio-tests)
endif()

if(STREAMGLASS_BUILD_TESTS)
    add_executable(
        streamglass-playback-tests tests/native/test_playback_queue.cpp
    )
    target_link_libraries(streamglass-playback-tests PRIVATE streamglass_audio)
    target_compile_options(
        streamglass-playback-tests
        PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
    add_test(NAME streamglass-playback-tests COMMAND streamglass-playback-tests)

    add_executable(
        streamglass-segmenter-tests tests/native/test_utterance_segmenter.cpp
    )
    target_link_libraries(streamglass-segmenter-tests PRIVATE streamglass_audio)
    target_compile_options(
        streamglass-segmenter-tests
        PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
    add_test(NAME streamglass-segmenter-tests COMMAND streamglass-segmenter-tests)
endif()
