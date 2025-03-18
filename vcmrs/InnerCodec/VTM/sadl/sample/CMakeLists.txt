cmake_minimum_required(VERSION 3.5)

project(sample LANGUAGES CXX)

set(SPARSE_MATMULT_SUPPORT OFF CACHE BOOL "add support for sparse matrix multiplication" )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (SPARSE_MATMULT_SUPPORT)
 add_definitions(-DSPARSE_SUPPORT=1) 
 message("Build with sparse matrix multiplication support")
endif()

include_directories(..)
file(GLOB HEADER_FILES helper.h ../sadl/*.h )

add_executable(sample_generic   sample.cpp ${HEADER_FILES})
if( NOT (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64") )
  add_executable(sample_simd256   sample.cpp ${HEADER_FILES})
  add_executable(sample_simd512   sample.cpp ${HEADER_FILES})
  set_target_properties(sample_simd256  PROPERTIES COMPILE_FLAGS "-DNDEBUG=1 " )
  set_target_properties(sample_simd512  PROPERTIES COMPILE_FLAGS "-DNDEBUG=1" )
endif()

add_executable(count_mac   count_mac.cpp ${HEADER_FILES})
set_target_properties(count_mac  PROPERTIES COMPILE_FLAGS "-DNDEBUG=1 " )  # must build in scalar mode to count MAC

add_executable(debug_model   debug_model.cpp ${HEADER_FILES})

add_executable(naive_quantization   naive_quantization.cpp ${HEADER_FILES} dumper.h)

add_executable(update_model_version update_model_version.cpp ${HEADER_FILES} dumper.h)

add_executable(quantization_test quantization_test.cpp ${HEADER_FILES})

if( UNIX OR MINGW )
 set(CMAKE_CXX_FLAGS_RELEASE "-O3")
 set(CMAKE_CXX_FLAGS "-ffast-math -Wall -fstrict-aliasing")
 if( NOT (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64") )
   set_target_properties(sample_simd256  PROPERTIES COMPILE_FLAGS "-mavx2" )
   set_target_properties(sample_simd512  PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512dq" )
   set_target_properties(debug_model  PROPERTIES COMPILE_FLAGS "-mavx2" ) # must build in SIMD mode to debug SIMD issue
 endif()
endif()


if( MSVC )
  set(CMAKE_CXX_FLAGS_RELEASE "/O2")
  set_target_properties(sample_simd256 PROPERTIES COMPILE_FLAGS "/arch:AVX2")
  set_target_properties(sample_simd512 PROPERTIES COMPILE_FLAGS "/arch:AVX512")
  set_target_properties(debug_model PROPERTIES COMPILE_FLAGS "/arch:AVX2")
  string( APPEND CMAKE_CXX_FLAGS " /WX" ) #treat warnings as errors
endif()
