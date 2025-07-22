#include <cuda_runtime.h>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
#include <ff/pasta.hpp>
#include "../sppark/msm/pippenger.cuh"
#include "../sppark/msm/batch_addition.cuh"

using point_p_t     = jacobian_t<pallas_t>;
using affine_t      = typename xyzz_t<pallas_t>::affine_t;
using affine_mem_t  = typename affine_t::mem_t;  
using scalar_in     = pallas_t;
using scalar_out    = pallas_t;

__device__ pallas_t make_beta() {
    return pallas_t(beta_c);
}

__shared__ pallas_t β_shared;

__global__ void glv_split_kernel(
    const scalar_in *in,
    scalar_out      *k1,
    scalar_out      *k2,
    size_t           npoints
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npoints) return;

    const uint32_t *kl = reinterpret_cast<const uint32_t*>(in + idx);
    uint32_t o1[8], o2[8];
    glv_split(kl, o1, o2);

    uint32_t *d1 = reinterpret_cast<uint32_t*>(k1 + idx);
    uint32_t *d2 = reinterpret_cast<uint32_t*>(k2 + idx);
    #pragma unroll
    for(int j=0;j<8;j++){
        d1[j] = o1[j];
        d2[j] = o2[j];
    }
}

__global__ void psi_kernel(
    const affine_mem_t *in,
    affine_mem_t       *out,
    size_t              npoints
) {
    if (threadIdx.x == 0)
        β_shared = make_beta();
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npoints) return;

    affine_t p = in[idx];
    p.X *= β_shared;
    out[idx] = p;
}
extern "C" void cuda_pippenger_pallas_glv(
    point_p_t           *out,
    const affine_mem_t  *h_points,
    size_t               npoints,
    const scalar_in     *h_scalars,
    bool                 mont
) {
    scalar_in *d_scalars;
    cudaMalloc(&d_scalars, sizeof(scalar_in) * npoints);
    scalar_out *d_k1, *d_k2;
    cudaMalloc(&d_k1, sizeof(scalar_out) * npoints);
    cudaMalloc(&d_k2, sizeof(scalar_out) * npoints);

    affine_mem_t *d_pts, *d_psi;
    cudaMalloc(&d_pts, sizeof(affine_mem_t) * npoints);
    cudaMalloc(&d_psi, sizeof(affine_mem_t) * npoints);

    point_p_t R1, R2;

    cudaMemcpy(d_scalars, h_scalars, sizeof(scalar_in) * npoints, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pts,     h_points,  sizeof(affine_mem_t) * npoints, cudaMemcpyHostToDevice);

    size_t grid = (npoints + 255) / 256;
    glv_split_kernel<<<grid, 256>>>(d_scalars, d_k1, d_k2, npoints);
    cudaDeviceSynchronize();

    psi_kernel<<<grid, 256>>>(d_pts, d_psi, npoints);
    cudaDeviceSynchronize();

    mult_pippenger<xyzz_t<pallas_t>>(&R1, d_pts, npoints, d_k1, true);
    mult_pippenger<xyzz_t<pallas_t>>(&R2, d_psi, npoints, d_k2, true);

    *out = R1;
    out->add(R2);

    cudaFree(d_scalars);
    cudaFree(d_k1);
    cudaFree(d_k2);
    cudaFree(d_pts);
    cudaFree(d_psi);
}