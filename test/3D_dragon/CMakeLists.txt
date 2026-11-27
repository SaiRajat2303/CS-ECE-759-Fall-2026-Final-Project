cmake_minimum_required(VERSION 3.24)
project(bvh_dynamic_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# Dependencies: header-only libraries that you download yourself, before
# configuring, by running ./prefetch.sh (it fills _deps/; versions are pinned
# there). CMake never downloads anything.
#   tinyobjloader - OBJ loading
#   happly        - PLY loading/saving (Stanford 3D Scanning Repository format)---------------------------------------------------------------------------
set(DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_deps" CACHE PATH "Folder filled by ./prefetch.sh")
foreach(header tinyobjloader/tiny_obj_loader.h happly/happly.h)
  if(NOT EXISTS "${DEPS_DIR}/${header}")
    message(FATAL_ERROR "Missing ${DEPS_DIR}/${header}\n"
                        "CMake does not fetch dependencies: run ./prefetch.sh in ${CMAKE_CURRENT_SOURCE_DIR} "
                        "manually first (on a node with internet; it only downloads source code).")
  endif()
endforeach()

add_library(mesh_deps INTERFACE)
target_include_directories(mesh_deps INTERFACE
  ${DEPS_DIR}/tinyobjloader
  ${DEPS_DIR}/happly
  ${CMAKE_CURRENT_SOURCE_DIR})


# ---------------------------------------------------------------------------
# CPU tools (always built)
# ---------------------------------------------------------------------------
add_executable(make_frames make_frames.cpp)
target_link_libraries(make_frames PRIVATE mesh_deps)


# ---------------------------------------------------------------------------
# CUDA harness (built only if a CUDA compiler is found)
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)   # override: -DCMAKE_CUDA_ARCHITECTURES=86
  endif()
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # The LBVH lives in the repository's include/ and src/ folders. Its Morton
  # and hierarchy kernels come from src/kernels.cu; the triangle-specific
  # kernels are in include/lbvh/tri_impl.h, wrapped by lbvh_user_bvh.cuh.
  set(LBVH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "LBVH repository root")
  set(LBVH_DEFS LBVH_BLOCK_SIZE=256 LBVH_LAYOUT_SOA=0 LBVH_STREAM_HINT_CG=0)

  add_executable(bvh_cuda_bench bvh_cuda_bench.cu ${LBVH_ROOT}/src/kernels.cu)
  target_include_directories(bvh_cuda_bench PRIVATE ${LBVH_ROOT}/include)
  target_compile_definitions(bvh_cuda_bench PRIVATE ${LBVH_DEFS})
  target_link_libraries(bvh_cuda_bench PRIVATE mesh_deps CUDA::cudart)
  target_compile_options(bvh_cuda_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)
else()
  message(WARNING "No CUDA compiler found: building make_frames only")
endif()