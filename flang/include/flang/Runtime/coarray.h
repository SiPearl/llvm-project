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

// FIXME: Improving this part with a compiler flag later ?
// Unimplemented prif runtime functions
#define DECLARE_UNIMPLEMENTED_PRIF(func_name, ...)                      \
  void _QMprifPprif_##func_name(__VA_ARGS__) {                                            \
    std::fprintf(stderr, "unimplemented feature : prif_" #func_name "\n"); \
    std::exit(EXIT_FAILURE);                                              \
  }

DECLARE_UNIMPLEMENTED_PRIF(init, void *)
DECLARE_UNIMPLEMENTED_PRIF(stop, void *, void*, void*)
DECLARE_UNIMPLEMENTED_PRIF(num_images, void *)
DECLARE_UNIMPLEMENTED_PRIF(num_images_with_team, void *, void *)
DECLARE_UNIMPLEMENTED_PRIF(num_images_with_team_number, void *, void *)

} // namespace Fortran::runtime
FORTRAN_EXTERN_C_END

#endif //FORTRAN_RUNTIME_PRIF_H_

