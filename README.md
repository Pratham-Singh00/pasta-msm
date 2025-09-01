## Mage: MSM Acceleration via GLV Enhancements

Mage is a performance-oriented library for arguments of knowledge, based on the pasta-msm implementation from Supranational and inspired by the sppark library. The library focuses on accelerating one of the most computationally expensive components of zero-knowledge proof generation: multi-scalar multiplication (MSM). Mage provides a Rust implementation for the Pasta curves (Pallas and Vesta), leveraging Gallant-Lambert-Vanstone (GLV) enhancements to achieve significant speedups.

---

## Step 1: Prerequisites

1. **Clone the Repository:**
	Clone the provided source code repository to your local machine.

2. **Install Rustup:**
	Open a terminal and run the following commands to install the Rust toolchain and configure the shell environment:
	```sh
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
	. "$HOME/.cargo/env"
	```

---

## Step 2: Modify Dependencies

Two modifications are required to run the benchmarks: one to a cached dependency and one to this project's configuration file.

1. **Modify the semolina Crate:**
	The benchmark requires access to private fields within the semolina crate. This necessitates a manual edit of the cached source code.
	- First, navigate to the semolina crate's source directory in the local cargo registry:
	  ```sh
	  cd ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/semolina-0.1.4/
	  ```
	  *Note: The hash `1949cf8c6b5b557f` may differ by system. Please navigate to `~/.cargo/registry/src` to find the correct path for `semolina-0.1.4`.*
	- Next, locate the file containing the `Affine_t` struct definition. The `field_t X, Y;` declarations must be moved under the `public:` access specifier.

	**Original Code:**
	```cpp
	class Affine_t {
		field_t X, Y;
	public:
		inline __host__ __device__ Affine_t() {}
		inline __host__ __device__ Affine_t(const field_t& x, const field_t& y) :
														X(x),            Y(y) {}
	};
	```

	**Modified Code:**
	```cpp
	class Affine_t {
	public:
		field_t X, Y;
		inline __host__ __device__ Affine_t() {}
		inline __host__ __device__ Affine_t(const field_t& x, const field_t& y) :
														X(x),            Y(y) {}
	};
	```

2. **Update Cargo.toml:**
	Return to the root directory of this project. Open the `Cargo.toml` file and replace the `[dev-dependencies]` section with the following:
	```toml
	[dev-dependencies]
	criterion = { version = "0.3", features = [ "html_reports" ] }
	rand = { version = "0.8", features = ["std", "small_rng"] }
	rand_chacha = "=0.3.1"
	rayon = "1.5"
	```
3. **Change numbers for 3090 or Jetson**
   
   If the code is running on the 3090, leave code as is.
   
   If the Code is running on the Jetson, do the following.
   
   On **pippenger.cuh**
   1. Change the ACCUMULATE_NTHREADS value from 384 to 512 (change 384 to 512 value on line 232 and 235)
   2. Change the MSM_NTHREADS value from 256 to 128 on line 240
      
   On **batch_addition.cuh**
   
   3. Replace batch_addition.cuh with the following:
   ```cuh
   // Copyright Supranational LLC
   // Licensed under the Apache License, Version 2.0, see LICENSE for details.
   // SPDX-License-Identifier: Apache-2.0

   #ifndef __SPPARK_MSM_BATCH_ADDITION_CUH__
   #define __SPPARK_MSM_BATCH_ADDITION_CUH__

   #include <cuda.h>
   #include <cooperative_groups.h>
   #include <vector>

   #include <ff/shfl.cuh>

   #ifndef WARP_SZ
   # define WARP_SZ 64
   #endif 
   #define TILE_SIZE 512
   #define BATCH_ADD_BLOCK_SIZE 256
   #define COARSENING_FACTOR 4
   #ifndef BATCH_ADD_NSTREAMS
   # define BATCH_ADD_NSTREAMS 8
   #elif BATCH_ADD_NSTREAMS == 0
   # error "invalid BATCH_ADD_NSTREAMS"
   #endif

   template<class bucket_t, class affine_h,
            class bucket_h = class bucket_t::mem_t,
            class affine_t = class bucket_t::affine_t>
   __device__ __forceinline__
   static void add(bucket_h ret[], const affine_h points[], uint32_t npoints,
                   const uint32_t bitmap[], const uint32_t refmap[],
                   bool accumulate, uint32_t sid)
   {

       static __device__ uint32_t streams[BATCH_ADD_NSTREAMS];
       uint32_t& current = streams[sid % BATCH_ADD_NSTREAMS];

       const uint32_t degree = bucket_t::degree;
       const uint32_t warp_sz = WARP_SZ / degree;
       const uint32_t tid = (threadIdx.x + blockDim.x*blockIdx.x) / degree;
       const uint32_t xid_base = (tid % warp_sz) * COARSENING_FACTOR;

       uint32_t laneid;
       asm("mov.u32 %0, %laneid;" : "=r"(laneid));

       bucket_t accs[COARSENING_FACTOR];
       #pragma unroll
       for (int i = 0; i < COARSENING_FACTOR; i++) {
           accs[i].inf();
           }

       if (accumulate && tid < gridDim.x*blockDim.x/WARP_SZ) {
           #pragma unroll
           for (int i = 0; i < COARSENING_FACTOR; i++) {
               accs[i] = ret[tid];
               }
       }

       uint32_t base = laneid == 0 ? atomicAdd(&current, 32*WARP_SZ) : 0;
       base = __shfl_sync(0xffffffff, base, 0);

       uint32_t chunk = min(32*WARP_SZ, npoints - base);
       uint32_t bits = 0, refs = 0, word = 0, sign = 0;
       uint32_t offs[COARSENING_FACTOR] = {0xffffffff, 0xffffffff};

       for (uint32_t i = 0, j = 0; base < npoints;) {
           if (i == 0) {
               bits = bitmap[base/WARP_SZ + laneid];
               refs = refmap ? refmap[base/WARP_SZ + laneid] : 0;

               bits ^= refs;
               refs &= bits;
           }

           for (; i < chunk && j < warp_sz; i++) {
               if (i % 32 == 0)
                   word = __shfl_sync(0xffffffff, bits, i/32);
               if (refmap && (i % 32 == 0))
                   sign = __shfl_sync(0xffffffff, refs, i/32);

               if (word & 1) {
                   #pragma unroll
                   for (int c = 0; c < COARSENING_FACTOR; c++) {
                       if (j == xid_base + c) {
                           offs[c] = (base + i) | (sign << 31);
                       }
                   }
                   j++;
               }
               word >>= 1;
               sign >>= 1;
           }

           if (i == chunk) {
               base = laneid == 0 ? atomicAdd(&current, 32*WARP_SZ) : 0;
               base = __shfl_sync(0xffffffff, base, 0);
               chunk = min(32*WARP_SZ, npoints - base);
               i = 0;
           }

           if (base >= npoints || j == warp_sz) {
               #pragma unroll
               for (int c = 0; c < COARSENING_FACTOR; c++) {
                   if (offs[c] != 0xffffffff) {
                       affine_t p = points[offs[c] & 0x7fffffff];
                       if (degree == 2) {
                           accs[c].uadd(p, offs[c] >> 31);
                       } else {
                           accs[c].add(p, offs[c] >> 31);
                           }
                       offs[c] = 0xffffffff;
                   }
               }
               j = 0;
           }
       }

       bucket_t acc = accs[0];
       #pragma unroll
       for (int i = 1; i < COARSENING_FACTOR; i++) {
           acc.uadd(accs[i]);
           }
   
       for (uint32_t off = 1; off < warp_sz * COARSENING_FACTOR;) {
           auto down = shfl_down(acc, off * degree);
           off <<= 1;
           if (((xid_base / COARSENING_FACTOR) & (off - 1)) == 0) {
               acc.uadd(down);
              }
       }

       cooperative_groups::this_grid().sync();

       if ((xid_base / COARSENING_FACTOR) == 0) {
           ret[tid / (warp_sz * COARSENING_FACTOR)] = acc;
           }

       if (threadIdx.x + blockIdx.x == 0) {
           current = 0;
           }
   }

   template<class bucket_t, class affine_h,
            class bucket_h = class bucket_t::mem_t,
            class affine_t = class bucket_t::affine_t>
   __launch_bounds__(BATCH_ADD_BLOCK_SIZE) __global__
   void batch_addition(bucket_h ret[], const affine_h points[], uint32_t npoints,
                       const uint32_t bitmap[], bool accumulate = false,
                       uint32_t sid = 0)
   {   add<bucket_t>(ret, points, npoints, bitmap, nullptr, accumulate, sid);   }

   template<class bucket_t, class affine_h,
            class bucket_h = class bucket_t::mem_t,
            class affine_t = class bucket_t::affine_t>
   __launch_bounds__(BATCH_ADD_BLOCK_SIZE) __global__
   void batch_diff(bucket_h ret[], const affine_h points[], uint32_t npoints,
                   const uint32_t bitmap[], const uint32_t refmap[],
                   bool accumulate = false, uint32_t sid = 0)
   {   add<bucket_t>(ret, points, npoints, bitmap, refmap, accumulate, sid);   }

   template<class bucket_t, class affine_h,
            class bucket_h = class bucket_t::mem_t,
            class affine_t = class bucket_t::affine_t>
   __launch_bounds__(BATCH_ADD_BLOCK_SIZE) __global__
   void batch_addition(bucket_h ret[], const affine_h points[], size_t npoints,
                       const uint32_t digits[], const uint32_t& ndigits)
   {
       const uint32_t degree = bucket_t::degree;
       const uint32_t warp_sz = WARP_SZ / degree;
       const uint32_t tid = (threadIdx.x + blockDim.x*blockIdx.x) / degree;
       const uint32_t xid = tid % warp_sz;

       bucket_t acc;
       acc.inf();

       __shared__ affine_t point_tile[TILE_SIZE];

      int tids = threadIdx.x + blockIdx.x * blockDim.x;

       for (size_t tile_start = 0; tile_start < ndigits; tile_start += TILE_SIZE) {
       // Load tile of points into shared memory
       if (threadIdx.x < TILE_SIZE && tile_start + threadIdx.x < ndigits) {
           uint32_t digit = digits[tile_start + threadIdx.x];
           point_tile[threadIdx.x] = points[digit & 0x7fffffff];
       }

       __syncthreads();

       // Iterate over this tile using thread striding
       for (size_t i = tile_start + tids; i < tile_start + TILE_SIZE && i < ndigits; i += gridDim.x * blockDim.x / degree) {
           uint32_t digit = digits[i];
           affine_t p = point_tile[i - tile_start];

           if (degree == 2)
               acc.uadd(p, digit >> 31);
           else
               acc.add(p, digit >> 31);
       }

       __syncthreads();
   }


       for (uint32_t off = 1; off < warp_sz;) {
           auto down = shfl_down(acc, off*degree);

           off <<= 1;
           if ((xid & (off-1)) == 0)
               acc.uadd(down); // .add() triggers spills ... in .shfl_down()
       }

       if (xid == 0)
           ret[tid/warp_sz] = acc;
   }

   template<class bucket_t>
   bucket_t sum_up(const bucket_t inp[], size_t n)
   {
       bucket_t sum = inp[0];
       for (size_t i = 1; i < n; i++)
           sum.add(inp[i]);
       return sum;
   }

   template<class bucket_t>
   bucket_t sum_up(const std::vector<bucket_t>& inp)
   {   return sum_up(&inp[0], inp.size());   }
   #endif
     
   ```
---
 
## Step 3: Run the Benchmark for Time

With the environment configured, the benchmarks can be executed.

1. **Clean Previous Builds (Recommended):**
	```sh
	cargo clean
	```

2. **Run the Benchmark Suite:**
	This command compiles the project in release mode and executes the benchmark tests.
	```sh
	cargo bench
	```

	Upon completion, detailed HTML reports are generated in the `target/criterion/report/index.html` directory.

---

## Step 4: Power Benchmark

The time measurement is done using the `cargo bench` command. The power measurement is based on the following two scripts:

### For RTX 3090 Ti
```sh
rm -f gpu.log
nvidia-smi --query-gpu=index,timestamp,power.draw --format=csv,noheader,nounits -lms 100 > gpu.log &
PID_NSMI=$!
./target/release/deps/main-dedb5e40bf5d9cb0
kill $PID_NSMI
```

### For Jetson
```sh
rm -f tegra.log
sudo tegrastats --interval 100 --logfile tegra.log &
PID=$!
./target/release/deps/main-0ef136a5bb36b405
sudo kill $PID
```

---

## (Optional) Step 5: GPU Profiling with NVIDIA NCU

These instructions are for obtaining a profile using NVIDIA Nsight Compute (`ncu`). This procedure is intended for environments with a compatible NVIDIA GPU and CUDA toolkit installed.

1. **Identify the ncu executable path:**
	```sh
	which ncu
	```

2. **Navigate to the build directory containing the target executable, referred to here as `main_exec`.**

3. **Execute the profiler:**
	Replace `<ncu-full-path>` with the path obtained in the first step.
	```sh
	sudo <ncu-full-path> --set full -f --export test-run.ncu-rep ./main_exec 17
	```

	This command generates a profile report file named `test-run.ncu-rep`.
