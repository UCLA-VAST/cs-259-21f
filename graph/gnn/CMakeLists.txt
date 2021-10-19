cmake_minimum_required(VERSION 3.14)

project(gnn)

set(PLATFORM
    xilinx_aws-vu9p-f1_shell-v04261818_201920_2
    CACHE STRING "Target FPGA platform")

find_package(TAPA REQUIRED)
find_package(SDx REQUIRED)

add_executable(gnn)
target_sources(gnn PRIVATE gnn-host.cpp gnn.cpp)
target_link_libraries(gnn PRIVATE tapa::tapa)

add_tapa_target(
  hls
  INPUT gnn.cpp
  TOP Gnn
  PLATFORM ${PLATFORM})

add_xocc_hw_link_targets(
  ${CMAKE_CURRENT_BINARY_DIR}
  INPUT hls
  HW_EMU_XCLBIN hw_emu_xclbin
  HW_XCLBIN hw_xclbin)

add_custom_target(
  swsim
  COMMAND $<TARGET_FILE:gnn> ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS gnn
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(
  hwsim
  COMMAND TAPAB=$<TARGET_PROPERTY:${hw_emu_xclbin},FILE_NAME> $<TARGET_FILE:gnn>
          ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS gnn ${hw_emu_xclbin}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(
  hw
  COMMAND TAPAB=$<TARGET_PROPERTY:${hw_xclbin},FILE_NAME> $<TARGET_FILE:gnn>
          ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS gnn ${hw_xclbin}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
