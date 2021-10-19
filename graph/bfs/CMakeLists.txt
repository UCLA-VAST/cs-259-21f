cmake_minimum_required(VERSION 3.14)

project(bfs)

set(PLATFORM
    xilinx_aws-vu9p-f1_shell-v04261818_201920_2
    CACHE STRING "Target FPGA platform")

find_package(TAPA REQUIRED)
find_package(SDx REQUIRED)

add_executable(bfs)
target_sources(bfs PRIVATE nxgraph.hpp bfs-host.cpp bfs.cpp)
target_link_libraries(bfs PRIVATE tapa::tapa)
file(
  DOWNLOAD "https://snap.stanford.edu/data/facebook_combined.txt.gz"
  ${CMAKE_CURRENT_BINARY_DIR}/facebook.txt.gz
  EXPECTED_HASH
    "SHA512=54b65fcdd41cad3e96e52f7ca60f326bea117cc8997ad4ec1831053fafa067fae9e728916b81a1fcf41a24de19e416b78bbfb14a3824c6e2bf67704dd7f89ad3"
)
execute_process(
  INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/facebook.txt.gz
  COMMAND gzip -cd
  OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/facebook.txt)

add_tapa_target(
  hls
  INPUT bfs.cpp
  TOP Bfs
  PLATFORM ${PLATFORM})

add_xocc_hw_link_targets(
  ${CMAKE_CURRENT_BINARY_DIR}
  INPUT hls
  HW_EMU_XCLBIN hw_emu_xclbin
  HW_XCLBIN hw_xclbin)

add_custom_target(
  swsim
  COMMAND $<TARGET_FILE:bfs> ${CMAKE_CURRENT_BINARY_DIR}/facebook.txt 512
  DEPENDS bfs)
add_custom_target(
  hwsim
  COMMAND TAPAB=$<TARGET_PROPERTY:${hw_emu_xclbin},FILE_NAME> $<TARGET_FILE:bfs>
          ${CMAKE_CURRENT_SOURCE_DIR}/graph.txt
  DEPENDS bfs ${hw_emu_xclbin})
add_custom_target(
  hw
  COMMAND TAPAB=$<TARGET_PROPERTY:${hw_xclbin},FILE_NAME> $<TARGET_FILE:bfs>
          ${CMAKE_CURRENT_BINARY_DIR}/facebook.txt
  DEPENDS bfs ${hw_xclbin})
