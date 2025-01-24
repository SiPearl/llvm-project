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

