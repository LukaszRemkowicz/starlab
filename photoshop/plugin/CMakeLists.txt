cmake_minimum_required(VERSION 3.20)
project(StarEffectsLab CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PHOTOSHOP_SDK "${CMAKE_SOURCE_DIR}/../sdk/mac/pluginsdk" CACHE PATH "Adobe Photoshop SDK root (pluginsdk)")
if(APPLE)
  set(_default_cmd "/Users/lukaszremkowicz/Projects/starlab/.venv/bin/star-effects-lab")
  set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "" FORCE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "")
else()
  set(_default_cmd "C:/Program Files/Star Effects Lab/star-effects-lab.exe")
endif()
# shortcut: dev default; the installer will set the real launcher path.
set(STARLAB_COMMAND "${_default_cmd}" CACHE STRING "Absolute path of the Star Effects Lab launcher")

add_library(selb_bridge STATIC bridge.cpp)
if(APPLE)
  target_link_libraries(selb_bridge PUBLIC "-framework CoreFoundation")
endif()

# Self-test: C++ <-> Python protocol interop, no Photoshop needed.
add_executable(bridge_selftest bridge_selftest.cpp)
target_link_libraries(bridge_selftest selb_bridge)

if(EXISTS "${PHOTOSHOP_SDK}/photoshopapi/photoshop")
  set(_sdk_includes
    "${PHOTOSHOP_SDK}/photoshopapi/photoshop"
    "${PHOTOSHOP_SDK}/photoshopapi/pica_sp"
    "${PHOTOSHOP_SDK}/photoshopapi"
    "${PHOTOSHOP_SDK}/samplecode/common/includes")
  set(_plugin_sources plugin.cpp bridge.cpp)
  string(REPLACE "\\" "/" _cmd "${STARLAB_COMMAND}")

  if(APPLE)
    add_library(StarEffectsLabPlugin MODULE ${_plugin_sources})
    set_target_properties(StarEffectsLabPlugin PROPERTIES
      BUNDLE TRUE
      BUNDLE_EXTENSION "plugin"
      OUTPUT_NAME "Star Effects Lab"
      # Own folder: Photoshop follows a symlinked folder (like RC-Astro's) but not a symlinked bundle.
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Star Effects Lab"
      MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist.in"
      RESOURCE "${CMAKE_CURRENT_SOURCE_DIR}/PiPLs.json")
    target_link_libraries(StarEffectsLabPlugin PRIVATE "-framework CoreFoundation")
    # Ad-hoc signature (CLT only, no Xcode): re-sign after the bundle is complete.
    add_custom_command(TARGET StarEffectsLabPlugin POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_BUNDLE_CONTENT_DIR:StarEffectsLabPlugin>/Resources"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/PiPLs.json"
              "$<TARGET_BUNDLE_CONTENT_DIR:StarEffectsLabPlugin>/Resources/PiPLs.json"
      # Binary PiPL too: Photoshop versions before the JSON PiPL (e.g. 2026) only read <name>.rsrc.
      COMMAND Rez -useDF -d __PIMac__=1 -d PRAGMA_ONCE=0
              -i "${PHOTOSHOP_SDK}/photoshopapi/resources" -i "${PHOTOSHOP_SDK}/photoshopapi/photoshop"
              -i "${PHOTOSHOP_SDK}/photoshopapi/pica_sp"
              -o "$<TARGET_BUNDLE_CONTENT_DIR:StarEffectsLabPlugin>/Resources/Star Effects Lab.rsrc"
              "${CMAKE_CURRENT_SOURCE_DIR}/PiPL.r"
      COMMAND codesign --force -s - "$<TARGET_BUNDLE_DIR:StarEffectsLabPlugin>"
      VERBATIM)
  else()
    add_library(StarEffectsLabPlugin MODULE ${_plugin_sources} StarEffectsLab.rc)
    set_target_properties(StarEffectsLabPlugin PROPERTIES OUTPUT_NAME "StarEffectsLab" SUFFIX ".8bf" PREFIX "")
  endif()
  target_include_directories(StarEffectsLabPlugin PRIVATE ${_sdk_includes})
  target_compile_definitions(StarEffectsLabPlugin PRIVATE STARLAB_COMMAND="${_cmd}")
else()
  message(WARNING "Photoshop SDK not found at ${PHOTOSHOP_SDK}; building bridge_selftest only")
endif()
