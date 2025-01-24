//===-- Coarray.h -- generate Coarray intrinsics runtime calls --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COARRAY_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COARRAY_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "flang/Lower/AbstractConverter.h"

namespace fir {
  class ExtendedValue;
  class FirOpBuilder;
} // namespace fir

namespace fir::runtime {

// Get the function type for a prif subroutine with a variable number of
// arguments
#define PRIF_FUNCTYPE(...)        \
   mlir::FunctionType::get(builder.getContext(),      \
                           /*inputs*/{__VA_ARGS__}, /*result*/{})

// Default prefix for subroutines of PRIF compiled with LLVM
#define PRIFTYPE_PREFIX "_QM__prifT"
#define PRIFTYPE(fmt) []() { \
  std::ostringstream oss;         \
  oss << PRIFTYPE_PREFIX << fmt;      \
  return oss.str();               \
}()

// Default prefix for subroutines of PRIF compiled with LLVM
#define PRIFSUB_PREFIX "_QMprifPprif_"
#define PRIFNAME_SUB(fmt) []() { \
  std::ostringstream oss;         \
  oss << PRIFSUB_PREFIX << fmt;      \
  return oss.str();               \
}()

/// Generate Call to runtime prif_num_images
mlir::Value getNumImages(fir::FirOpBuilder &builder, mlir::Location loc);

/// Generate Call to runtime prif_num_images_with_team or 
/// prif_num_images_with_team_number
mlir::Value getNumImagesWithTeam(fir::FirOpBuilder &builder, mlir::Location loc,
                   mlir::Value team);

} // fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COARRAY_H


