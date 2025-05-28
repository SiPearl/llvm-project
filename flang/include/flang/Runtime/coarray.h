//===-- include/flang/Runtime/prif.h -----------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_PRIF_H_
#define FORTRAN_RUNTIME_PRIF_H_

#include "flang/Common/ISO_Fortran_binding_wrapper.h"
#include "flang/Runtime/c-or-cpp.h"
#include "flang/Runtime/entry-names.h"
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#include <vector>

FORTRAN_EXTERN_C_BEGIN

namespace Fortran::runtime {
class Descriptor;

void RTNAME(ComputeLastUcobound)(
    int num_images, Descriptor &lcobounds, Descriptor &ucobounds);

void RTNAME(Copy1DArrayToI64Array)(
    const Descriptor &from, const Descriptor &to);
} // namespace Fortran::runtime

FORTRAN_EXTERN_C_END

#endif // FORTRAN_RUNTIME_PRIF_H_
