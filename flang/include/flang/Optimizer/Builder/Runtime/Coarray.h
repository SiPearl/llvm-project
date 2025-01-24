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

/// Generate call to runtime function that store prif_coarray_handle with addr
void saveCoarrayHandle(fir::FirOpBuilder &builder, mlir::Location loc,
                       mlir::Value addr, mlir::Value handle);

/// Generate call to runtime function to retrieve prif_coarray_handle
/// associated to an addr
mlir::Value getCoarrayHandle(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value addr);

/// Generate call to runtime function to compute the lastest ucobound.
void computeLastUcobound(fir::FirOpBuilder &builder, mlir::Location loc,
                         mlir::Value lcobounds, mlir::Value ucobounds);

void copy1DArrayToI64Array(fir::FirOpBuilder &builder, mlir::Location loc,
                           mlir::Value from, mlir::Value to);

/// Generate Call to runtime prif_num_images
mlir::Value getNumImages(fir::FirOpBuilder &builder, mlir::Location loc);

/// Generate Call to runtime prif_num_images_with_team or 
/// prif_num_images_with_team_number
mlir::Value getNumImagesWithTeam(fir::FirOpBuilder &builder, mlir::Location loc,
                   mlir::Value team);

/// Generate Call to runtime prif_this_image_no_coarray
mlir::Value getThisImage(fir::FirOpBuilder &builder, mlir::Location loc,
                         mlir::Value team = {});

/// Generate Call to runtime prif_this_image_with_coarray or
/// prif_this_image_with_dim
mlir::Value getThisImageWithCoarray(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Type resultType,
                                    mlir::Value coarrayHandle, mlir::Value team,
                                    mlir::Value dim = {});

/// Generate Call to runtime prif_this_image_index
mlir::Value getImageIndex(fir::FirOpBuilder &builder, mlir::Location loc,
                          mlir::Value handle, mlir::Value sub,
                          mlir::Value team = {});

/// Generate Call to runtime prif_lcobound_{with|no}_dim
mlir::Value genLCoBounds(fir::FirOpBuilder &builder, mlir::Location loc,
                         mlir::Value handle, size_t corank,
                         mlir::Value dim = {});

/// Generate Call to runtime prif_ucobound_{with|no}_dim
mlir::Value genUCoBounds(fir::FirOpBuilder &builder, mlir::Location loc,
                         mlir::Value handle, size_t corank,
                         mlir::Value dim = {});

/// Generate call to runtime subroutine prif_sync_all
void genSyncAllStatement(fir::FirOpBuilder &builder, mlir::Location loc,
                         mlir::Value stat, mlir::Value errmsg);
/// Generate call to runtime subroutine prif_sync_memory
void genSyncMemoryStatement(fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Value stat, mlir::Value errmsg);
/// Generate call to runtime subroutine prif_sync_images
void genSyncImagesStatement(fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Value imageSet, mlir::Value stat,
                            mlir::Value errmsg);

/// Generate call to runtime subroutine prif_lock
void genLockStatement(fir::FirOpBuilder &builder, mlir::Location loc,
                      mlir::Value imageNum, mlir::Value lockVarAddr,
                      mlir::Value acquiredLock, mlir::Value offset,
                      mlir::Value stat, mlir::Value errmsg);
/// Generate call to runtime subroutine prif_unlock
void genUnlockStatement(fir::FirOpBuilder &builder, mlir::Location loc,
                        mlir::Value imageNum, mlir::Value lockVarAddr,
                        mlir::Value offset, mlir::Value stat,
                        mlir::Value errmsg);

/// Generate Call to runtime prif_fail_image
void genFailImageStatement(fir::FirOpBuilder &builder, mlir::Location loc);

/// Generate call to runtime subroutine prif_co_broadcast
void genCoBroadcast(fir::FirOpBuilder &builder, mlir::Location loc,
                    mlir::Value A, mlir::Value sourceImage, mlir::Value stat,
                    mlir::Value errmsg);

/// Generate call to runtime subroutine prif_co_max and prif_co_max_character
void genCoMax(fir::FirOpBuilder &builder, mlir::Location loc, mlir::Value A,
              mlir::Value resultImage, mlir::Value stat, mlir::Value errmsg);

/// Generate call to runtime subroutine prif_co_min or prif_co_min_character
void genCoMin(fir::FirOpBuilder &builder, mlir::Location loc, mlir::Value A,
              mlir::Value resultImage, mlir::Value stat, mlir::Value errmsg);

/// Generate call to runtime subroutine prif_co_sum_
void genCoSum(fir::FirOpBuilder &builder, mlir::Location loc, mlir::Value A,
              mlir::Value resultImage, mlir::Value stat, mlir::Value errmsg);

} // fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COARRAY_H


