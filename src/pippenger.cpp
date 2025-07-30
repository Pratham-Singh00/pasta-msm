// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// #include <msm/pippenger.hpp>
#include "../sppark/msm/pippenger.hpp"
#include <ec/affine_t.hpp>
#include "../sppark/msm/pippenger_common.hpp"
#include "../sppark/msm/pippenger_old.hpp"

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
#include <ff/pasta.hpp>
static thread_pool_t da_pool;


extern "C"
void mult_pippenger_pallas(jacobian_t<pallas_t>& ret,
                           const xyzz_t<pallas_t>::affine_t points[],
                           size_t npoints, const vesta_t scalars[], bool mont)
{   mult_pippenger<xyzz_t<pallas_t>>(ret, points, npoints, scalars, mont,
                                     &da_pool);
}


/* extern "C"
void mult_pippenger_vesta(
    jacobian_t<vesta_t>               &ret,
    const xyzz_t<vesta_t>::affine_t   pts[],
    size_t                            npts,
    const pallas_t                    scalars[],
    bool                              mont
) {
   
} */