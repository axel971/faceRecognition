
cmake_minimum_required( VERSION 2.6 FATAL_ERROR )
 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#SET(Boost_ADDITIONAL_VERSIONS "1.54.0" "1.55.0" )

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/module)

find_package(Boost COMPONENTS  filesystem system)
 
link_directories (${Boost_LIBRARY_DIRS} )
include_directories ( ${Boost_INCLUDE_DIRS} )
 
SET (SOURCES
source/main.cpp
)
 
SET (EXECUTABLE_NAME
executable
)
 
add_executable (
    ${EXECUTABLE_NAME}
    ${SOURCES}
    )
 
target_link_libraries (
    ${EXECUTABLE_NAME}
    ${Boost_LIBRARIES}
    )
