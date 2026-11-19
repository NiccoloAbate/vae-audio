cmake_minimum_required(VERSION 3.22)
project(LatentDrumMachine VERSION 0.1.0)

# Latent Drum Machine: a JUCE instrument that renders drum hits from a VAE latent
# recipe (kits exported by ../export_kit.py) and plays them as a sampler.
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
#
# Builds VST3 / AU / Standalone (copied to ~/Library/Audio/Plug-Ins) and
# kit_render_test, a command-line check of the kit + layer pipeline.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "libtorch from pip is arm64 only")
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "")

# ── LibTorch: the one inside the Python torch install (as CelloVAESynth) ────────
if(NOT TORCH_CMAKE_PATH)
    execute_process(
        COMMAND python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake'), end='')"
        OUTPUT_VARIABLE PYTHON_TORCH_PATH ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(TORCH_CMAKE_PATH "${PYTHON_TORCH_PATH}")
endif()
if(NOT TORCH_CMAKE_PATH)
    message(FATAL_ERROR "Set -DTORCH_CMAKE_PATH=<site-packages>/torch/share/cmake")
endif()
set(TORCH_CMAKE_PATH "${TORCH_CMAKE_PATH}" CACHE PATH "TorchConfig.cmake directory")
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
find_package(Torch REQUIRED)
message(STATUS "LibTorch: ${TORCH_CMAKE_PATH}")

# ── JUCE ──────────────────────────────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        8.0.9
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(JUCE)

# Default kit, loaded when a new instance starts
set(LDM_DEFAULT_KIT "${CMAKE_CURRENT_SOURCE_DIR}/../kits/v3_B1/kit.pt" CACHE FILEPATH "kit.pt loaded by default")
configure_file(src/BuildConfig.h.in "${CMAKE_CURRENT_BINARY_DIR}/generated/BuildConfig.h")

set(LDM_CORE_SOURCES src/KitModel.cpp src/Layers.cpp)

function(ldm_common target)
    target_include_directories(${target} PRIVATE src "${CMAKE_CURRENT_BINARY_DIR}/generated" ${TORCH_INCLUDE_DIRS})
    target_compile_definitions(${target} PUBLIC JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)
    target_compile_options(${target} PRIVATE -Wno-deprecated-declarations)
    target_link_libraries(${target} PRIVATE "${TORCH_LIBRARIES}")
endfunction()

# ── Plugin ────────────────────────────────────────────────────────────────────
juce_add_plugin(LatentDrumMachine
    VERSION                   0.1.0
    COMPANY_NAME              "CCRMA"
    PLUGIN_MANUFACTURER_CODE  Ccra
    PLUGIN_CODE               Ldm1
    FORMATS                   VST3 AU Standalone
    PRODUCT_NAME              "Latent Drum Machine"
    IS_SYNTH                  TRUE
    NEEDS_MIDI_INPUT          TRUE
    NEEDS_MIDI_OUTPUT         FALSE
    IS_MIDI_EFFECT            FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD   TRUE
    VST3_CATEGORIES           "Instrument" "Drum"
    AU_MAIN_TYPE              kAudioUnitType_MusicDevice)

target_sources(LatentDrumMachine PRIVATE
    ${LDM_CORE_SOURCES} src/RenderWorker.cpp src/Sampler.cpp
    src/PluginProcessor.cpp src/PluginEditor.cpp)
target_compile_definitions(LatentDrumMachine PUBLIC JUCE_VST3_CAN_REPLACE_VST2=0)
target_link_libraries(LatentDrumMachine PRIVATE juce::juce_audio_utils
    PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_lto_flags)
ldm_common(LatentDrumMachine)

# ── Command-line check of the kit + layer pipeline ───────────────────────────
juce_add_console_app(KitRenderTest PRODUCT_NAME "kit_render_test")
target_sources(KitRenderTest PRIVATE ${LDM_CORE_SOURCES} tools/RenderTest.cpp)
target_link_libraries(KitRenderTest PRIVATE juce::juce_audio_formats
    PUBLIC juce::juce_recommended_config_flags)
ldm_common(KitRenderTest)
