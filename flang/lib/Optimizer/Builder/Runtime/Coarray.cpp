//===-- Coarray.cpp -- runtime API for coarray intrinsics -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Coarray.h"
#include "flang/Optimizer/Builder/Runtime/Coarray.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Runtime/coarray.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace Fortran::runtime;
using namespace Fortran::semantics;

/// Test if an ExtendedValue is absent.
static bool isStaticallyAbsent(const fir::ExtendedValue &exv) {
  return !fir::getBase(exv);
}

/// Generate call to runtime function that store prif_coarray_handle with addr
void fir::runtime::saveCoarrayHandle(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value addr,
                                     mlir::Value handle) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(saveCoarrayHandle)>(loc, builder);

  mlir::Value refHandle = builder.create<fir::BoxAddrOp>(
      loc, builder.getRefType(handle.getType()), handle);
  llvm::SmallVector<mlir::Value> args = {addr, refHandle};
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to runtime function to retrieve prif_coarray_handle
/// associated to an addr
mlir::Value fir::runtime::getCoarrayHandle(fir::FirOpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::Value addr) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(getCoarrayHandle)>(loc, builder);

  llvm::SmallVector<mlir::Value> args = {addr};
  // return builder.create<fir::CallOp>(loc, func, args).getResult(0);
  return builder.createBox(
      loc, builder.create<fir::CallOp>(loc, func, args).getResult(0));
}

/// Generate call to runtime function to compute the lastest ucobound.
void fir::runtime::computeLastUcobound(fir::FirOpBuilder &builder,
                                       mlir::Location loc,
                                       mlir::Value lcobounds,
                                       mlir::Value ucobounds) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(computeLastUcobound)>(loc, builder);
  mlir::Value num_images = fir::runtime::getNumImages(builder, loc);
  llvm::SmallVector<mlir::Value> args = {num_images, lcobounds, ucobounds};
  builder.create<fir::CallOp>(loc, func, args);
}

void fir::runtime::copy1DArrayToI64Array(fir::FirOpBuilder &builder,
                                         mlir::Location loc, mlir::Value from,
                                         mlir::Value to) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(copy1DArrayToI64Array)>(loc,
                                                                   builder);
  llvm::SmallVector<mlir::Value> args = {from, to};
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate Call to runtime prif_num_images
mlir::Value fir::runtime::getNumImages(fir::FirOpBuilder &builder, mlir::Location loc) {
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
          
  mlir::Value result = builder.create<fir::AllocaOp>(loc, builder.getI32Type());
  mlir::FunctionType ftype = PRIF_FUNCTYPE(ptrTy);
  mlir::func::FuncOp funcOp = builder.createFunction(loc, PRIFNAME_SUB("num_images"), ftype);
  llvm::SmallVector<mlir::Value> args = {result};
  builder.create<fir::CallOp>(loc, funcOp, args);
  return builder.create<fir::LoadOp>(loc, result);
}

mlir::Value fir::runtime::getNumImagesWithTeam(fir::FirOpBuilder &builder, mlir::Location loc,
                   mlir::Value team) {
  std::string numImagesName = fir::unwrapPassByRefType(team.getType()).isInteger() ?
    PRIFNAME_SUB("num_images_with_team_number") : PRIFNAME_SUB("num_images_with_team");

  mlir::Value result = builder.create<fir::AllocaOp>(loc, builder.getI32Type());
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::FunctionType ftype = PRIF_FUNCTYPE(ptrTy, ptrTy);
  mlir::func::FuncOp funcOp = builder.createFunction(loc, numImagesName, ftype);
  llvm::SmallVector<mlir::Value> args = {team, result};
  builder.create<fir::CallOp>(loc, funcOp, args);
  return builder.create<fir::LoadOp>(loc, result);

}

/// Generate Call to runtime prif_this_image_no_coarray
mlir::Value fir::runtime::getThisImage(fir::FirOpBuilder &builder,
                                       mlir::Location loc, mlir::Value team) {
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::FunctionType ftype = PRIF_FUNCTYPE(ptrTy, ptrTy);
  mlir::func::FuncOp funcOp =
      builder.createFunction(loc, PRIFNAME_SUB("this_image_no_coarray"), ftype);

  mlir::Value result = builder.create<fir::AllocaOp>(loc, builder.getI32Type());
  mlir::Value teamArg =
      !isStaticallyAbsent(team)
          ? team
          : builder.create<fir::AbsentOp>(
                loc,
                fir::BoxType::get(mlir::NoneType::get(builder.getContext())));
  llvm::SmallVector<mlir::Value> args = {teamArg, result};
  builder.create<fir::CallOp>(loc, funcOp, args);
  return builder.create<fir::LoadOp>(loc, result);
}

/// Generate Call to runtime prif_this_image_with_coarray or
/// prif_this_image_with_dim
mlir::Value fir::runtime::getThisImageWithCoarray(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Type resultType,
    mlir::Value coarrayHandle, mlir::Value team, mlir::Value dim) {
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  llvm::SmallVector<mlir::Value> args;
  mlir::FunctionType ftype;
  mlir::func::FuncOp funcOp;
  mlir::Value result;
  if (!isStaticallyAbsent(dim)) {
    result = builder.create<fir::AllocaOp>(loc, resultType);
    ftype = PRIF_FUNCTYPE(ptrTy, ptrTy, ptrTy, ptrTy);
    funcOp =
        builder.createFunction(loc, PRIFNAME_SUB("this_image_with_dim"), ftype);
    args.insert(args.end(), {coarrayHandle, dim});
  } else {
    // Need to embox the array
    result = builder.createBox(loc, builder.createTemporary(loc, resultType));
    ftype = PRIF_FUNCTYPE(ptrTy, ptrTy, ptrTy);
    funcOp = builder.createFunction(
        loc, PRIFNAME_SUB("this_image_with_coarray"), ftype);
    args.push_back(coarrayHandle);
  }

  args.insert(args.end(), {team, result});
  builder.create<fir::CallOp>(loc, funcOp, args);
  return builder.create<fir::LoadOp>(loc, result);
}

/// Generate call to runtime prif_this_image_index and assumed that sub is
/// an array of i64 elements
mlir::Value fir::runtime::getImageIndex(fir::FirOpBuilder &builder,
                                        mlir::Location loc, mlir::Value handle,
                                        mlir::Value sub, mlir::Value team) {
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Value result = builder.create<fir::AllocaOp>(loc, builder.getI32Type());

  mlir::func::FuncOp funcOp;
  llvm::SmallVector<mlir::Value> localArgs = {handle, sub};
  if (isStaticallyAbsent(team)) {
    mlir::FunctionType ftype = PRIF_FUNCTYPE(ptrTy, ptrTy, ptrTy);
    funcOp = builder.createFunction(loc, PRIFNAME_SUB("image_index"), ftype);
    localArgs.emplace_back(result);
  } else {
    std::string imageIndexName =
        fir::unwrapPassByRefType(team.getType()).isInteger()
            ? PRIFNAME_SUB("image_index_with_team")
            : PRIFNAME_SUB("image_index_with_team_number");
    mlir::FunctionType ftype = PRIF_FUNCTYPE(ptrTy, ptrTy, ptrTy, ptrTy);
    funcOp = builder.createFunction(loc, imageIndexName, ftype);
    localArgs.insert(localArgs.end(), {team, result});
  }
  builder.create<fir::CallOp>(loc, funcOp, localArgs);
  return builder.create<fir::LoadOp>(loc, result);
}

/// Generate Call to runtime prif_lcobound_{with|no}_dim
mlir::Value fir::runtime::genLCoBounds(fir::FirOpBuilder &builder,
                                       mlir::Location loc, mlir::Value handle,
                                       size_t corank, mlir::Value dim) {
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Type resultType = fir::SequenceType::get(
      static_cast<fir::SequenceType::Extent>(corank), builder.getI64Type());
  mlir::Value result =
      builder.createBox(loc, builder.createTemporary(loc, resultType));

  mlir::func::FuncOp funcOp;
  llvm::SmallVector<mlir::Value> localArgs = {handle};
  if (isStaticallyAbsent(dim)) {
    mlir::FunctionType ftype = PRIF_FUNCTYPE(ptrTy, ptrTy);
    funcOp =
        builder.createFunction(loc, PRIFNAME_SUB("lcobound_no_dim"), ftype);
    localArgs.emplace_back(result);
  } else {
    mlir::FunctionType ftype = PRIF_FUNCTYPE(ptrTy, ptrTy, ptrTy);
    funcOp =
        builder.createFunction(loc, PRIFNAME_SUB("lcobound_with_dim"), ftype);
    localArgs.insert(localArgs.end(), {dim, result});
  }
  builder.create<fir::CallOp>(loc, funcOp, localArgs);
  return result;
}

/// Generate Call to runtime prif_ucobound_{with|no}_dim
mlir::Value fir::runtime::genUCoBounds(fir::FirOpBuilder &builder,
                                       mlir::Location loc, mlir::Value handle,
                                       size_t corank, mlir::Value dim) {
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Type resultType = fir::SequenceType::get(
      static_cast<fir::SequenceType::Extent>(corank), builder.getI64Type());
  mlir::Value result =
      builder.createBox(loc, builder.createTemporary(loc, resultType));

  mlir::func::FuncOp funcOp;
  llvm::SmallVector<mlir::Value> localArgs = {handle};
  if (isStaticallyAbsent(dim)) {
    mlir::FunctionType ftype = PRIF_FUNCTYPE(ptrTy, ptrTy);
    funcOp =
        builder.createFunction(loc, PRIFNAME_SUB("ucobound_no_dim"), ftype);
    localArgs.emplace_back(result);
  } else {
    mlir::FunctionType ftype = PRIF_FUNCTYPE(ptrTy, ptrTy, ptrTy);
    funcOp =
        builder.createFunction(loc, PRIFNAME_SUB("ucobound_with_dim"), ftype);
    localArgs.insert(localArgs.end(), {dim, result});
  }
  builder.create<fir::CallOp>(loc, funcOp, localArgs);
  return result;
}

/// Generate Call to runtime prif_coshape
mlir::Value fir::runtime::genCoshape(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value handle,
                                     size_t corank) {
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Type resultType = fir::SequenceType::get(
      static_cast<fir::SequenceType::Extent>(corank), builder.getI64Type());
  mlir::Value result =
      builder.createBox(loc, builder.createTemporary(loc, resultType));

  mlir::FunctionType ftype = PRIF_FUNCTYPE(ptrTy, ptrTy);
  mlir::func::FuncOp funcOp =
      builder.createFunction(loc, PRIFNAME_SUB("coshape"), ftype);
  llvm::SmallVector<mlir::Value> localArgs = {handle, result};
  builder.create<fir::CallOp>(loc, funcOp, localArgs);
  return result;
}

/// Generate call to runtime subroutine prif_sync_all
void fir::runtime::genSyncAllStatement(fir::FirOpBuilder &builder,
                                       mlir::Location loc, mlir::Value stat,
                                       mlir::Value errmsg) {
  mlir::Value nullPtr = builder.createNullConstant(loc);
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::FunctionType ftype = PRIF_FUNCTYPE(ptrTy, ptrTy, ptrTy);
  mlir::func::FuncOp funcOp =
      builder.createFunction(loc, PRIFNAME_SUB("sync_all"), ftype);

  llvm::SmallVector<mlir::Value> localArgs = {stat, errmsg, nullPtr};
  builder.create<fir::CallOp>(loc, funcOp, localArgs);
}

/// Generate call to runtime subroutine prif_sync_memory
void fir::runtime::genSyncMemoryStatement(fir::FirOpBuilder &builder,
                                          mlir::Location loc, mlir::Value stat,
                                          mlir::Value errmsg) {
  mlir::Value nullPtr = builder.createNullConstant(loc);
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::FunctionType ftype = PRIF_FUNCTYPE(ptrTy, ptrTy, ptrTy);
  mlir::func::FuncOp funcOp =
      builder.createFunction(loc, PRIFNAME_SUB("sync_memory"), ftype);

  llvm::SmallVector<mlir::Value> localArgs = {stat, errmsg, nullPtr};
  builder.create<fir::CallOp>(loc, funcOp, localArgs);
}

/// Generate call to runtime subroutine prif_sync_images
void fir::runtime::genSyncImagesStatement(fir::FirOpBuilder &builder,
                                          mlir::Location loc,
                                          mlir::Value imageSet,
                                          mlir::Value stat,
                                          mlir::Value errmsg) {
  mlir::Value nullPtr = builder.createNullConstant(loc);
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::FunctionType ftype = PRIF_FUNCTYPE(ptrTy, ptrTy, ptrTy, ptrTy);
  mlir::func::FuncOp funcOp =
      builder.createFunction(loc, PRIFNAME_SUB("sync_images"), ftype);

  llvm::SmallVector<mlir::Value> localArgs = {imageSet, stat, errmsg, nullPtr};
  builder.create<fir::CallOp>(loc, funcOp, localArgs);
}

/// Generate call to runtime subroutine prif_lock
void fir::runtime::genLockStatement(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value imageNum,
                                    mlir::Value lockVarAddr,
                                    mlir::Value acquiredLock,
                                    mlir::Value offset, mlir::Value stat,
                                    mlir::Value errmsg) {
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::FunctionType ftype =
      PRIF_FUNCTYPE(ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy);
  mlir::func::FuncOp funcOp =
      builder.createFunction(loc, PRIFNAME_SUB("lock"), ftype);

  mlir::Value nullPtr = builder.createNullConstant(loc);
  mlir::Value handle = getCoarrayHandle(builder, loc, lockVarAddr);
  llvm::SmallVector<mlir::Value> localArgs = {
      imageNum, handle, acquiredLock, offset, stat, errmsg, nullPtr};
  builder.create<fir::CallOp>(loc, funcOp, localArgs);
}

/// Generate call to runtime subroutine prif_unlock
void fir::runtime::genUnlockStatement(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value imageNum,
                                      mlir::Value lockVarAddr,
                                      mlir::Value offset, mlir::Value stat,
                                      mlir::Value errmsg) {
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::FunctionType ftype =
      PRIF_FUNCTYPE(ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy);
  mlir::func::FuncOp funcOp =
      builder.createFunction(loc, PRIFNAME_SUB("unlock"), ftype);

  mlir::Value nullPtr = builder.createNullConstant(loc);
  mlir::Value handle = getCoarrayHandle(builder, loc, lockVarAddr);
  llvm::SmallVector<mlir::Value> localArgs = {imageNum, handle, offset,
                                              stat,     errmsg, nullPtr};
  builder.create<fir::CallOp>(loc, funcOp, localArgs);
}

/// Generate Call to runtime prif_fail_image
void fir::runtime::genFailImageStatement(fir::FirOpBuilder &builder,
                                         mlir::Location loc) {
  mlir::FunctionType ftype =
      mlir::FunctionType::get(builder.getContext(), {}, {});
  mlir::func::FuncOp funcOp =
      builder.createFunction(loc, PRIFNAME_SUB("fail_image"), ftype);
  builder.create<fir::CallOp>(loc, funcOp);
}

/// Generate call to collective subroutines except co_reduce_
/// A must be lowered as a box
void genCollectiveSubroutine(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value A, mlir::Value sourceImage,
                             mlir::Value stat, mlir::Value errmsg,
                             std::string coName) {
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::FunctionType ftype = PRIF_FUNCTYPE(ptrTy, ptrTy, ptrTy, ptrTy, ptrTy);
  mlir::func::FuncOp funcOp = builder.createFunction(loc, coName, ftype);

  mlir::Value nullPtr = builder.createNullConstant(loc);
  llvm::SmallVector<mlir::Value> localArgs = {A, sourceImage, stat, nullPtr,
                                              errmsg};
  builder.create<fir::CallOp>(loc, funcOp, localArgs);
}
/// Generate call to runtime subroutine prif_co_broadcast
void fir::runtime::genCoBroadcast(fir::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::Value A,
                                  mlir::Value sourceImage, mlir::Value stat,
                                  mlir::Value errmsg) {
  genCollectiveSubroutine(builder, loc, A, sourceImage, stat, errmsg,
                          PRIFNAME_SUB("co_broadcast"));
}

/// Generate call to runtime subroutine prif_co_max or prif_co_max_character
void fir::runtime::genCoMax(fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Value A, mlir::Value resultImage,
                            mlir::Value stat, mlir::Value errmsg) {
  if (fir::unwrapPassByRefType(A.getType()).isInteger(8)) {
    // FIXME: Need to embox A into a CharBoxValue or CharArrayBoxValue ?
    genCollectiveSubroutine(builder, loc, A, resultImage, stat, errmsg,
                            PRIFNAME_SUB("co_max_character"));
  } else {
    genCollectiveSubroutine(builder, loc, A, resultImage, stat, errmsg,
                            PRIFNAME_SUB("co_max"));
  }
}

/// Generate call to runtime subroutine prif_co_min or prif_co_min_character
void fir::runtime::genCoMin(fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Value A, mlir::Value resultImage,
                            mlir::Value stat, mlir::Value errmsg) {
  if (fir::unwrapPassByRefType(A.getType()).isInteger(8)) {
    // FIXME: Need to embox A into a CharBoxValue or CharArrayBoxValue ?
    genCollectiveSubroutine(builder, loc, A, resultImage, stat, errmsg,
                            PRIFNAME_SUB("co_min_character"));
  } else {
    genCollectiveSubroutine(builder, loc, A, resultImage, stat, errmsg,
                            PRIFNAME_SUB("co_min"));
  }
}

/// Generate call to runtime subroutine prif_co_sum_
void fir::runtime::genCoSum(fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Value A, mlir::Value resultImage,
                            mlir::Value stat, mlir::Value errmsg) {
  genCollectiveSubroutine(builder, loc, A, resultImage, stat, errmsg,
                          PRIFNAME_SUB("co_sum"));
}
