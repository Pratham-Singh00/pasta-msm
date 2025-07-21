// cuda/pippenger_glv.cu

#define SPPARK_DONT_INSTANTIATE_TEMPLATES
#include "../sppark/msm/pippenger.cuh"

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
#include <ff/pasta.hpp>

extern "C" void mult_pippenger_pallas_glv(
    jacobian_t<pallas_t> &ret,
    const xyzz_t<pallas_t>::affine_t *points,
    size_t npoints,
    const vesta_t *scalars
);

__global__ void glv_split_kernel(
    const vesta_t *in,
    vesta_t       *k1,
    vesta_t       *k2,
    size_t         n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t kl[8], o1[8], o2[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        kl[j] = in[i][j];
    }
    glv_split(kl, o1, o2);

    vesta_t t1, t2;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        t1[j] = o1[j];
        t2[j] = o2[j];
    }
    k1[i] = t1;
    k2[i] = t2;
}

__global__ void psi_kernel(
    const xyzz_t<pallas_t>::affine_t *in,
          xyzz_t<pallas_t>::affine_t *out,
    size_t                            n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = in[i].psi();
}

extern "C" void mult_pippenger_pallas_glv(
    jacobian_t<pallas_t> &ret,
    const xyzz_t<pallas_t>::affine_t *points,
    size_t npoints,
    const vesta_t *scalars
) {
    vesta_t *d_k1, *d_k2;
    xyzz_t<pallas_t>::affine_t *d_psi;
    cudaMalloc(&d_k1,  npoints * sizeof(vesta_t));
    cudaMalloc(&d_k2,  npoints * sizeof(vesta_t));
    cudaMalloc(&d_psi, npoints * sizeof(*d_psi));

    size_t B = (npoints + 255) / 256;
    glv_split_kernel<<<B,256>>>(scalars, d_k1, d_k2, npoints);
    psi_kernel      <<<B,256>>>(points,  d_psi,       npoints);
    cudaDeviceSynchronize();

    jacobian_t<pallas_t> R1, R2;
    mult_pippenger<xyzz_t<pallas_t>>(&R1, points, npoints, d_k1);
    mult_pippenger<xyzz_t<pallas_t>>(&R2, d_psi,  npoints, d_k2);

    ret = R1;
    ret.add(R2);

    cudaFree(d_k1);
    cudaFree(d_k2);
    cudaFree(d_psi);
}
