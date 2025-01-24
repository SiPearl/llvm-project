//===-- include/flang/Runtime/prif.h -----------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_PRIF_H_
#define FORTRAN_RUNTIME_PRIF_H_

#include "flang/ISO_Fortran_binding_wrapper.h"
#include "flang/Runtime/c-or-cpp.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/entry-names.h"
#include <map>
#include <stdlib.h>
#include <iostream>
#include <tuple>
#include <vector>

FORTRAN_EXTERN_C_BEGIN

namespace Fortran::runtime {

// Map each coarray_handle pointer to the coarray base_addr
// FIXME: Improving storing of the coarray handle
static std::map<void *, Descriptor &> map_coarray_handle;

void RTNAME(saveCoarrayHandle)(void *base_addr, Descriptor &coarray_handle);

Descriptor &RTNAME(getCoarrayHandle)(void *base_addr);

void RTNAME(computeLastUcobound)(
    int num_images, Descriptor &lcobounds, Descriptor &ucobounds);

} // namespace Fortran::runtime

// FIXME: Improving this part with a compiler flag later ?
// Unimplemented prif runtime functions
#define DECLARE_UNIMPLEMENTED_PRIF(func_name, ...)                      \
  void _QMprifPprif_##func_name(__VA_ARGS__) {                                            \
    std::fprintf(stderr, "unimplemented feature : prif_" #func_name "\n"); \
    std::exit(EXIT_FAILURE);                                              \
  }

DECLARE_UNIMPLEMENTED_PRIF(init, void *)
DECLARE_UNIMPLEMENTED_PRIF(stop, void *, void*, void*)
DECLARE_UNIMPLEMENTED_PRIF(this_image_no_coarray, void *, void *)
DECLARE_UNIMPLEMENTED_PRIF(this_image_with_coarray, void *, void *, void *)
DECLARE_UNIMPLEMENTED_PRIF(this_image_with_dim, void *, void *, void *, void *)
DECLARE_UNIMPLEMENTED_PRIF(num_images, void *)
DECLARE_UNIMPLEMENTED_PRIF(num_images_with_team, void *, void *)
DECLARE_UNIMPLEMENTED_PRIF(num_images_with_team_number, void *, void *)
DECLARE_UNIMPLEMENTED_PRIF(allocate_coarray, void *, void *, void *, void *,
    void *, void *, void *, void *, void *)
DECLARE_UNIMPLEMENTED_PRIF(sync_memory, void *, void *, void *)
DECLARE_UNIMPLEMENTED_PRIF(sync_all, void *, void *, void *)
DECLARE_UNIMPLEMENTED_PRIF(sync_images, void *, void *, void *, void *)
DECLARE_UNIMPLEMENTED_PRIF(sync_team, void *, void *, void *, void *)
DECLARE_UNIMPLEMENTED_PRIF(co_broadcast, void *, void *, void *, void *, void *)
DECLARE_UNIMPLEMENTED_PRIF(co_max, void *, void *, void *, void *, void *)
DECLARE_UNIMPLEMENTED_PRIF(
    co_max_character, void *, void *, void *, void *, void *)
DECLARE_UNIMPLEMENTED_PRIF(co_min, void *, void *, void *, void *, void *)
DECLARE_UNIMPLEMENTED_PRIF(
    co_min_character, void *, void *, void *, void *, void *)
DECLARE_UNIMPLEMENTED_PRIF(co_sum, void *, void *, void *, void *, void *)

FORTRAN_EXTERN_C_END

#endif //FORTRAN_RUNTIME_PRIF_H_

