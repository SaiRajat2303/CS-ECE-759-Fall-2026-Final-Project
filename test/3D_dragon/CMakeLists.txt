cmake_minimum_required(VERSION 3.22)
project(bvh_dynamic_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# Dependencies (downloaded at configure time)
#   tinyobjloader - OBJ loading
#   happly        - PLY loading/saving (Stanford 3D Scanning Repository format)
# Both are used header-only. SOURCE_SUBDIR points at a directory that does not
# exist so FetchContent only downloads them and does not add their own CMake
# projects (tinyobjloader's would build a second copy of the implementation).
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(tinyobjloader
  GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
  GIT_TAG        v2.0.0rc13
  GIT_SHALLOW    TRUE
  SOURCE_SUBDIR  _header_only)

FetchContent_Declare(happly
  GIT_REPOSITORY https://github.com/nmwsharp/happly.git
  GIT_TAG        v1.1.0
  GIT_SHALLOW    TRUE
  SOURCE_SUBDIR  _header_only)

FetchContent_MakeAvailable(tinyobjloader happly)

add_library(mesh_deps INTERFACE)
target_include_directories(mesh_deps INTERFACE
  ${tinyobjloader_SOURCE_DIR}
  ${happly_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR})

# ---------------------------------------------------------------------------
# CPU tools (always built)
# ---------------------------------------------------------------------------
add_executable(make_frames make_frames.cpp)
target_link_libraries(make_frames PRIVATE mesh_deps)

add_executable(bvh_bench bvh_dynamic_bench.cpp)   # CPU reference harness
target_include_directories(bvh_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

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

  add_executable(bvh_cuda_bench bvh_cuda_bench.cu)
  target_link_libraries(bvh_cuda_bench PRIVATE mesh_deps CUDA::cudart)
  target_compile_options(bvh_cuda_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)
else()
  message(WARNING "No CUDA compiler found: building CPU tools only (make_frames, bvh_bench)")
endif()