#!/usr/bin/env vivado_hls
source params.tcl
open_project -reset ${project}
#add_files -cflags -std=c++11 ../graph-tlp-hls.cpp
add_files -cflags "-std=c++11 -DCOSIM" Graph.cpp
add_files -cflags -std=c++11 -tb graph-cosim.cpp
set_top ${top}
open_solution -reset ${solution}
set_part ${part}
create_clock -period ${period}
config_interface -m_axi_addr64
set_directive_interface -mode m_axi -offset slave -bundle num_vertices -depth 1024 ${top} num_vertices
set_directive_interface -mode m_axi -offset slave -bundle num_edges -depth 1024 ${top} num_edges
set_directive_interface -mode m_axi -offset slave -bundle vertices -depth 1024 ${top} vertices
set_directive_interface -mode m_axi -offset slave -bundle edges -depth 1024 ${top} edges
set_directive_interface -mode m_axi -offset slave -bundle updates -depth 1024 ${top} updates
csynth_design
close_project

#open_project -reset ${project}
#add_files -cflags -std=c++11 ../graph-tlp-hls.cpp
#set_top Graph
#open_solution -reset ${solution}
#set_part ${part}
#create_clock -period ${period}
#config_interface -m_axi_addr64
#csynth_design
#close_project
