cmake_minimum_required(VERSION 3.5)

set(BUILD_WLAN_INTERFACE OFF CACHE BOOL "Disable WiFi interface")
set(BUILD_ETHERNET_INTERFACE OFF CACHE BOOL "Disable Ethernet interface")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(microros_esp32)
