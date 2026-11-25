# 1. Get the dragon: dragon_recon.tar.gz from graphics.stanford.edu/data/3Dscanrep/

tar xzf dragon_recon.tar.gz # assets/graphics/dragon_vrip.ply (~871K tris); # dragon_vrip_res2/3/4.ply are smaller versions for quick tests

# 2. Build (downloads tinyobjloader + happly automatically)

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. Make animated frames from the static scan

./build/make_frames dragon_recon/dragon_vrip.ply explode 60 frames/explode
./build/make_frames dragon_recon/dragon_vrip.ply wave 60 frames/wave

# 4. Benchmark your BVH

./build/bvh*cuda_bench "frames/explode/frame*%04d.ply" 60 explode.csv

# Install Nsight Systems

Find the version corresponding to your OS here:

[Nsight](https://developer.nvidia.com/nsight-systems/get-started)

Download the generated `.nsys-rep` or .`ncu-rep` files to your local workstation using scp or rsync, then open them in the matching desktop version of Nsight Systems or Nsight Compute.
