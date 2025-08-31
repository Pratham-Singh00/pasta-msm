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
