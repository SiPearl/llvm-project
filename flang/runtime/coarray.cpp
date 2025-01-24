//===-- runtime/coarray.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/coarray.h"
#include "type-info.h"

namespace Fortran::runtime {

void RTNAME(saveCoarrayHandle)(void *base_addr, Descriptor &coarray_handle) {
  map_coarray_handle.emplace(
      std::pair<void *, Descriptor &>(base_addr, coarray_handle));
}

Descriptor &RTNAME(getCoarrayHandle)(void *base_addr) {
  auto it = map_coarray_handle.find(base_addr);
  if (it != map_coarray_handle.end())
    return it->second;
  return map_coarray_handle.begin()->second;
}

void RTNAME(computeLastUcobound)(
    int num_images, Descriptor &lcobounds, Descriptor &ucobounds) {
  int corank = ucobounds.GetDimension(0).Extent();
  int64_t *lcobounds_ptr = (int64_t *)lcobounds.raw().base_addr;
  int64_t *ucobounds_ptr = (int64_t *)ucobounds.raw().base_addr;
  int64_t index = 1;
  for (int i = 0; i < corank - 1; i++) {
    index *= ucobounds_ptr[i] - lcobounds_ptr[i] + 1;
  }
  if (index < num_images)
    ucobounds_ptr[corank - 1] =
        (num_images / index) + (num_images % index != 0);
}

void RTNAME(copy1DArrayToI64Array)(
    const Descriptor &from, const Descriptor &to) {
  void *from_ptr = from.raw().base_addr;
  int64_t *to_ptr = (int64_t *)to.raw().base_addr;
  int rank = from.raw().rank;
  for (int j = 0; j < from.raw().dim[0].extent; j++) {
    if (from.raw().type == CFI_type_int16_t)
      to_ptr[j] = ((int16_t *)from_ptr)[j];
    else if (from.raw().type == CFI_type_int32_t)
      to_ptr[j] = ((int32_t *)from_ptr)[j];
    else if (from.raw().type == CFI_type_int64_t)
      to_ptr[j] = ((int64_t *)from_ptr)[j];
    else if (from.raw().type == CFI_type_intmax_t)
      to_ptr[j] = ((intmax_t *)from_ptr)[j];
  }
}
} // namespace Fortran::runtime
