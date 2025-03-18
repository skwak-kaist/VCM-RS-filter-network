cmake_minimum_required(VERSION 3.5)

project(test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
file(GLOB HEADER_FILES ../sadl/*.h )

include_directories(..)
add_executable(test_scalar  test.cpp ${HEADER_FILES})
add_executable(test_avx2    test.cpp ${HEADER_FILES})
add_executable(test_avx512  test.cpp ${HEADER_FILES})
add_executable(test_border_scalar  test_border.cpp ${HEADER_FILES})
add_executable(test_border_avx2    test_border.cpp ${HEADER_FILES})

if( UNIX OR MINGW )
 set(CMAKE_CXX_FLAGS_RELEASE "-O3")
 set(CMAKE_CXX_FLAGS "-ffast-math -Wall -fstrict-aliasing") 
 set_target_properties(test_scalar PROPERTIES COMPILE_FLAGS "" )
 set_target_properties(test_avx2   PROPERTIES COMPILE_FLAGS "-mavx2" )
 set_target_properties(test_avx512 PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512dq" )
 set_target_properties(test_border_scalar PROPERTIES COMPILE_FLAGS "" )
 set_target_properties(test_border_avx2   PROPERTIES COMPILE_FLAGS "-mavx2" )
endif()


if( MSVC )
  set(CMAKE_CXX_FLAGS_RELEASE "/O2")
  set_target_properties(test_avx2 PROPERTIES COMPILE_FLAGS "/arch:AVX2")
  set_target_properties(test_border_avx2 PROPERTIES COMPILE_FLAGS "/arch:AVX2")
  string( APPEND CMAKE_CXX_FLAGS " /WX" ) #treat warnings as errors
endif()
