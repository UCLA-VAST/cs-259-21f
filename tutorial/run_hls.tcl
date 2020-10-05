############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 2013 Xilinx Inc. All rights reserved.
############################################################

# Create a Vivado HLS project
open_project -reset vadd_prj
set_top vadd
add_files vadd_baseline.cpp
# add_files vadd_lcs_step1.cpp
# add_files vadd_mem_step2.cpp
# add_files vadd_cmp_step3.cpp

# Solution1 *************************
open_solution -reset "solution1"
#set_part  {xc7k160tfbg484-2}
set_part {xcvu9p-flga2104-2L-e}
create_clock -period 10

# Run Synthesis
csynth_design

# Create the IP package
export_design

exit



