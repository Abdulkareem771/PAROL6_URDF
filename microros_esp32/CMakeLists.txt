cmake_minimum_required(VERSION 3.5)

# FORCE UART-only micro-ROS; must be set before project() / including IDF project files.
set(BUILD_WLAN_INTERFACE OFF CACHE BOOL "Disable WiFi interface" FORCE)
set(BUILD_ETHERNET_INTERFACE OFF CACHE BOOL "Disable Ethernet interface" FORCE)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(microros_esp32)
