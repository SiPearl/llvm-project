//===-- Coarray.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Implementation of the lowering of image related constructs and expressions.
/// Fortran images can form teams, communicate via coarrays, etc.
///
//===----------------------------------------------------------------------===//

#include "flang/Lower/Coarray.h"
#include "flang/Evaluate/fold.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Allocatable.h"
#include "flang/Optimizer/Builder/Runtime/Coarray.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Runtime/allocatable.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/runtime-type-info.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace Fortran::semantics;
using namespace Fortran::runtime;

/// Test if an ExtendedValue is absent.
static bool isStaticallyAbsent(const fir::ExtendedValue &exv) {
  return !fir::getBase(exv);
}

//===----------------------------------------------------------------------===//
// TEAM statements and constructs
//===----------------------------------------------------------------------===//

void Fortran::lower::genChangeTeamConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &,
    const Fortran::parser::ChangeTeamConstruct &) {
  TODO(converter.getCurrentLocation(), "coarray: CHANGE TEAM construct");
}

void Fortran::lower::genChangeTeamStmt(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &,
    const Fortran::parser::ChangeTeamStmt &stmt) {
  // TODO(converter.getCurrentLocation(), "coarray: CHANGE TEAM statement");
  mlir::Location loc = converter.getCurrentLocation();
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  mlir::Value errMsg, stat, team;
  // Handle STAT and ERRMSG values
  Fortran::lower::StatementContext stmtCtx;
  const std::list<Fortran::parser::StatOrErrmsg> &statOrErrList =
      std::get<std::list<Fortran::parser::StatOrErrmsg>>(stmt.t);
  for (const Fortran::parser::StatOrErrmsg &statOrErr : statOrErrList) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::StatVariable &statVar) {
              const auto *expr = Fortran::semantics::GetExpr(statVar);
              stat = fir::getBase(converter.genExprAddr(loc, *expr, stmtCtx));
            },
            [&](const Fortran::parser::MsgVariable &errMsgVar) {
              const auto *expr = Fortran::semantics::GetExpr(errMsgVar);
              errMsg = fir::getBase(converter.genExprBox(loc, *expr, stmtCtx));
            },
        },
        statOrErr.u);
  }

  if (isStaticallyAbsent(stat))
    stat = builder.create<fir::AbsentOp>(
        loc, builder.getRefType(builder.getI32Type()));
  if (isStaticallyAbsent(errMsg))
    errMsg = builder.create<fir::AbsentOp>(
        loc, fir::BoxType::get(mlir::NoneType::get(builder.getContext())));

  // Handle TEAM-VALUE
  const auto *teamExpr =
      Fortran::semantics::GetExpr(std::get<Fortran::parser::TeamValue>(stmt.t));
  team = fir::getBase(converter.genExprBox(loc, *teamExpr, stmtCtx));

  fir::runtime::genChangeTeamStatement(builder, loc, team, stat, errMsg);
}

void Fortran::lower::genEndChangeTeamStmt(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &,
    const Fortran::parser::EndChangeTeamStmt &stmt) {
  mlir::Location loc = converter.getCurrentLocation();
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  mlir::Value errMsg, stat;
  // Handle STAT and ERRMSG values
  Fortran::lower::StatementContext stmtCtx;
  const std::list<Fortran::parser::StatOrErrmsg> &statOrErrList =
      std::get<std::list<Fortran::parser::StatOrErrmsg>>(stmt.t);
  for (const Fortran::parser::StatOrErrmsg &statOrErr : statOrErrList) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::StatVariable &statVar) {
              const auto *expr = Fortran::semantics::GetExpr(statVar);
              stat = fir::getBase(converter.genExprAddr(loc, *expr, stmtCtx));
            },
            [&](const Fortran::parser::MsgVariable &errMsgVar) {
              const auto *expr = Fortran::semantics::GetExpr(errMsgVar);
              errMsg = fir::getBase(converter.genExprBox(loc, *expr, stmtCtx));
            },
        },
        statOrErr.u);
  }

  if (isStaticallyAbsent(stat))
    stat = builder.create<fir::AbsentOp>(
        loc, builder.getRefType(builder.getI32Type()));
  if (isStaticallyAbsent(errMsg))
    errMsg = builder.create<fir::AbsentOp>(
        loc, fir::BoxType::get(mlir::NoneType::get(builder.getContext())));

  fir::runtime::genEndTeamStatement(builder, loc, stat, errMsg);
}

void Fortran::lower::genFormTeamStatement(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &,
    const Fortran::parser::FormTeamStmt &stmt) {
  mlir::Location loc = converter.getCurrentLocation();
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  mlir::Value errMsg, stat, newIndex, teamNumber, team;
  // Handle NEW_INDEX, STAT and ERRMSG
  std::list<Fortran::parser::StatOrErrmsg> statOrErrList{};
  Fortran::lower::StatementContext stmtCtx;
  const auto &formSpecList =
      std::get<std::list<Fortran::parser::FormTeamStmt::FormTeamSpec>>(stmt.t);
  for (const Fortran::parser::FormTeamStmt::FormTeamSpec &formSpec :
       formSpecList) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::StatOrErrmsg &statOrErr) {
              std::visit(
                  Fortran::common::visitors{
                      [&](const Fortran::parser::StatVariable &statVar) {
                        const auto *expr = Fortran::semantics::GetExpr(statVar);
                        stat = fir::getBase(
                            converter.genExprAddr(loc, *expr, stmtCtx));
                      },
                      [&](const Fortran::parser::MsgVariable &errMsgVar) {
                        const auto *expr =
                            Fortran::semantics::GetExpr(errMsgVar);
                        errMsg = fir::getBase(
                            converter.genExprBox(loc, *expr, stmtCtx));
                      },
                  },
                  statOrErr.u);
            },
            [&](const Fortran::parser::ScalarIntExpr &intExpr) {
              fir::ExtendedValue newIndexExpr = converter.genExprValue(
                  loc, Fortran::semantics::GetExpr(intExpr), stmtCtx);
              newIndex = fir::getBase(newIndexExpr);
            },
        },
        formSpec.u);
  }
  if (isStaticallyAbsent(newIndex))
    newIndex = builder.create<fir::AbsentOp>(
        loc, builder.getRefType(builder.getI32Type()));
  if (isStaticallyAbsent(stat))
    stat = builder.create<fir::AbsentOp>(
        loc, builder.getRefType(builder.getI32Type()));
  if (isStaticallyAbsent(errMsg))
    errMsg = builder.create<fir::AbsentOp>(
        loc, fir::BoxType::get(mlir::NoneType::get(builder.getContext())));

  // Handle TEAM-NUMBER
  const auto *teamNumberExpr = Fortran::semantics::GetExpr(
      std::get<Fortran::parser::ScalarIntExpr>(stmt.t));
  teamNumber =
      fir::getBase(converter.genExprAddr(loc, *teamNumberExpr, stmtCtx));

  // Handle TEAM-VARIABLE
  const auto *teamExpr = Fortran::semantics::GetExpr(
      std::get<Fortran::parser::TeamVariable>(stmt.t));
  team = fir::getBase(converter.genExprBox(loc, *teamExpr, stmtCtx));

  fir::runtime::genFormTeamStatement(builder, loc, teamNumber, team, newIndex,
                                     stat, errMsg);
}

//===----------------------------------------------------------------------===//
// COARRAY utils 
//===----------------------------------------------------------------------===//

llvm::SmallVector<mlir::Value, 4>
Fortran::lower::genCoshape(Fortran::lower::AbstractConverter &converter,
                           mlir::Location loc,
                           const Fortran::lower::SomeExpr &expr,
                           Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  llvm::SmallVector<mlir::Value, 4> vecCoshape;

  auto *e = Fortran::evaluate::UnwrapExpr<Fortran::lower::SomeExpr>(expr);
  const Fortran::semantics::Symbol *sym = Fortran::evaluate::GetLastSymbol(*e);
  if (sym && sym->GetUltimate().has<Fortran::semantics::ObjectEntityDetails>()) {
    const auto *object{sym->GetUltimate().detailsIf<Fortran::semantics::ObjectEntityDetails>()};
    auto coshape = object->coshape();
    size_t corank = coshape.size();
    mlir::Type i64Ty = builder.getI64Type();

    for(size_t i = 0; i < corank; i++) {
      long int lcb = 1, ucb = -1; // default
      // Lower cobounds
      Fortran::semantics::Bound lcobound = coshape[i].lbound();
      if (lcobound.GetExplicit().has_value()) {
        auto b = Fortran::evaluate::ToInt64(lcobound.GetExplicit().value());
        if(b.has_value())
          lcb = b.value();
      }
      
      // Upper cobounds
      Fortran::semantics::Bound ucobound = coshape[i].ubound();
      if (ucobound.GetExplicit().has_value()) {
        auto b = Fortran::evaluate::ToInt64(ucobound.GetExplicit().value());
        if(b.has_value())
          ucb = b.value();
      }
      size_t coextent = ucb < lcb ? -1 : (ucb - lcb + 1);
      vecCoshape.push_back(builder.createIntegerConstant(loc, i64Ty, coextent));
    }
  }
  return vecCoshape;
}

llvm::SmallVector<mlir::Value, 4>
Fortran::lower::genCoSubscripts(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
            const Fortran::lower::SomeExpr &expr, Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  llvm::SmallVector<mlir::Value, 4> imageIndex;

  if (auto coarrayRef{Fortran::evaluate::ExtractCoarrayRef(expr)}) {
    const Fortran::semantics::Symbol sym = coarrayRef.value().GetLastSymbol();
    if (sym.GetUltimate().has<Fortran::semantics::ObjectEntityDetails>()) {
      const auto *object{sym.GetUltimate().detailsIf<Fortran::semantics::ObjectEntityDetails>()};
      size_t corank = coarrayRef.value().cosubscript().size();
      for (unsigned i = 0; i < corank; i++) {
        auto c = ignoreEvConvert(coarrayRef.value().cosubscript()[i]);
        imageIndex.push_back(fir::getBase(converter.genExprValue(loc, c, stmtCtx)));
      }
    }
  }
  return imageIndex;
}

std::pair<mlir::Value, mlir::Value>
Fortran::lower::genCoarrayCoBounds(Fortran::lower::AbstractConverter &converter,
                                   mlir::Location loc,
                                   const Fortran::semantics::Symbol &sym) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Value ucobounds, lcobounds;

  if (sym.GetUltimate().has<Fortran::semantics::ObjectEntityDetails>()) {
    const auto *object{
        sym.GetUltimate().detailsIf<Fortran::semantics::ObjectEntityDetails>()};
    auto coshape = object->coshape();
    size_t corank = coshape.size();
    mlir::Type i64Ty = builder.getI64Type();
    mlir::Type addrType = builder.getRefType(i64Ty);
    mlir::Type arrayType = fir::SequenceType::get(
        {static_cast<fir::SequenceType::Extent>(corank)}, i64Ty);
    lcobounds = builder.createTemporary(loc, arrayType);
    ucobounds = builder.createTemporary(loc, arrayType);

    for (size_t i = 0; i < corank; i++) {
      long int lcobound = 1, ucobound = -1; // default cobound value
      auto index =
          builder.createIntegerConstant(loc, builder.getIndexType(), i);
      // Lower cobounds
      Fortran::semantics::Bound lbound = coshape[i].lbound();
      if (lbound.GetExplicit().has_value()) {
        auto b = Fortran::evaluate::ToInt64(lbound.GetExplicit().value());
        if (b.has_value())
          lcobound = b.value();
      }
      auto lcovalue = builder.createIntegerConstant(loc, i64Ty, lcobound);
      auto lcoaddr =
          builder.create<fir::CoordinateOp>(loc, addrType, lcobounds, index);
      builder.create<fir::StoreOp>(loc, lcovalue, lcoaddr);

      // Upper cobounds
      Fortran::semantics::Bound ubound = coshape[i].ubound();
      if (ubound.GetExplicit().has_value()) {
        auto b = Fortran::evaluate::ToInt64(ubound.GetExplicit().value());
        if (b.has_value())
          ucobound = b.value();
      }
      auto ucovalue = builder.createIntegerConstant(loc, i64Ty, ucobound);
      auto ucoaddr =
          builder.create<fir::CoordinateOp>(loc, addrType, ucobounds, index);
      builder.create<fir::StoreOp>(loc, ucovalue, ucoaddr);
    }
    lcobounds = builder.createBox(loc, lcobounds);
    ucobounds = builder.createBox(loc, ucobounds);
  }
  return {lcobounds, ucobounds};
}

/// From cosubscript, generate call to runtime function prif_image_index
/// associated to an addr
mlir::Value Fortran::lower::getImageIndexFromCosubscripts(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const Fortran::evaluate::CoarrayRef &expr, mlir::Value handle) {
  // Creation of the cosubscripts array
  mlir::Type i64Ty = builder.getI64Type();
  unsigned corank = expr.cosubscript().size();
  mlir::Type indexType = builder.getIndexType();
  mlir::Type arrayType = fir::SequenceType::get(
      {static_cast<fir::SequenceType::Extent>(corank)}, i64Ty);
  mlir::Value cosubscripts = builder.createTemporary(loc, arrayType);
  mlir::Type addrType = builder.getRefType(i64Ty);
  for (unsigned dim = 0; dim < corank; ++dim) {
    auto image = ToInt64(expr.cosubscript()[dim]);
    mlir::Value idx;
    if (image.has_value())
      idx = builder.createIntegerConstant(loc, i64Ty, image.value());
    else {
      auto s = ignoreEvConvert(expr.cosubscript()[dim]);
      TODO(loc, "getting image_index with a non constant cosubscript.");
    }

    auto index = builder.createIntegerConstant(loc, indexType, dim);
    auto coAddr =
        builder.create<fir::CoordinateOp>(loc, addrType, cosubscripts, index);
    builder.create<fir::StoreOp>(loc, idx, coAddr);
  }
  cosubscripts = builder.createBox(loc, cosubscripts);

  // Computation of the image_index
  return fir::runtime::getImageIndex(builder, loc, handle, cosubscripts);
}

//===----------------------------------------------------------------------===//
// COARRAY memory management
//===----------------------------------------------------------------------===//

mlir::Type getCoarrayHandleType(fir::FirOpBuilder &builder,
                                mlir::Location loc) {
  // Defining the coarray handle type
  std::string handleDTName = PRIFTYPE("prif_coarray_handle");
  fir::RecordType handleTy =
      fir::RecordType::get(builder.getContext(), handleDTName);
  mlir::Type ptrTy = fir::LLVMPointerType::get(
      builder.getContext(),
      mlir::FunctionType::get(builder.getContext(), {}, {}));
  handleTy.finalize({}, {{"info", ptrTy}});

  // Checking if the type information was generated
  fir::TypeInfoOp dt;
  fir::RecordType parentType{};
  mlir::OpBuilder::InsertPoint insertPointIfCreated;
  std::tie(dt, insertPointIfCreated) =
      builder.createTypeInfoOp(loc, handleTy, parentType);
  if (insertPointIfCreated.isSet()) {
    // fir.type_info wasn't built in a previous call.
    dt->setAttr(dt.getNoInitAttrName(), builder.getUnitAttr());
    dt->setAttr(dt.getNoDestroyAttrName(), builder.getUnitAttr());
    dt->setAttr(dt.getNoFinalAttrName(), builder.getUnitAttr());
    builder.restoreInsertionPoint(insertPointIfCreated);
    // Create global op
    // FIXME: replace handleTy by the Derived type that describe handleTy
    std::string globalName =
        fir::NameUniquer::getTypeDescriptorName(handleDTName);
    auto linkage = builder.createLinkOnceODRLinkage();
    builder.createGlobal(loc, handleTy, globalName, linkage);
  }
  return handleTy;
}

static mlir::Value
genAllocateCoarrayRuntimeCall(fir::FirOpBuilder &builder, mlir::Location loc,
                              mlir::Type baseType, mlir::Value lcobounds,
                              mlir::Value ucobounds, mlir::Value allocMem,
                              mlir::Value stat, mlir::Value errMsg) {
  // Generate call to prif_allocate_coarray with the correct mangling name
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Type i64Ty = builder.getI64Type();
  mlir::FunctionType ftype = PRIF_FUNCTYPE(ptrTy, ptrTy, ptrTy, ptrTy, ptrTy,
                                           ptrTy, ptrTy, ptrTy, ptrTy);
  mlir::func::FuncOp funcOp =
      builder.createFunction(loc, PRIFNAME_SUB("allocate_coarray"), ftype);
  fir::runtime::computeLastUcobound(builder, loc, lcobounds, ucobounds);

  // Handle ELEMENT_SIZE
  std::optional<mlir::DataLayout> dl = fir::support::getOrSetDataLayout(
      builder.getModule(), /*allowDefaultLayout*/ true);
  mlir::Value elementSize = builder.createTemporary(loc, i64Ty);
  auto [size, align] = fir::getTypeSizeAndAlignmentOrCrash(
      loc, baseType, *dl, builder.getKindMap());
  builder.create<fir::StoreOp>(
      loc, builder.createIntegerConstant(loc, i64Ty, size), elementSize);

  // Allocate instance of prif_coarray_handle type based on the PRIF
  // specification.
  mlir::Type handleTy = getCoarrayHandleType(builder, loc);
  mlir::Value coarrayHandle =
      builder.createBox(loc, builder.createHeapTemporary(loc, handleTy));

  // TODO: Handle FINAL_FUNC argument ("coarray_cleanup")
  mlir::Value finalFunc = builder.createTemporary(loc, ptrTy);
  mlir::Value none = builder.create<fir::AbsentOp>(
      loc, fir::BoxType::get(mlir::NoneType::get(builder.getContext())));

  llvm::SmallVector<mlir::Value> args = {lcobounds, ucobounds,     elementSize,
                                         finalFunc, coarrayHandle, allocMem,
                                         stat,      none,          errMsg};
  builder.create<fir::CallOp>(loc, funcOp, args);
  return coarrayHandle;
}

/// Generate call to prif_allocate_coarray runtime subroutine based on
// a fir::MutableBoxValue
void Fortran::lower::genAllocateCoarray(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::semantics::Symbol &sym, fir::MutableBoxValue box) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  // Handle LCOBOUNDS and UCOBOUNDS form the Fortran::semantics::Symbol
  auto [lcobounds, ucobounds] =
      Fortran::lower::genCoarrayCoBounds(converter, loc, sym);

  // TODO: Handle STAT and ERRMSG
  mlir::Value stat = builder.create<fir::AbsentOp>(
      loc, builder.getRefType(builder.getI32Type()));
  mlir::Value errMsg = builder.create<fir::AbsentOp>(
      loc, fir::BoxType::get(mlir::NoneType::get(builder.getContext())));

  mlir::Value coarrayHandle = genAllocateCoarrayRuntimeCall(
      builder, loc, fir::getBaseTypeOf(box), lcobounds, ucobounds,
      fir::getBase(box), stat, errMsg);

  // Saving the coarray_handle.
  fir::runtime::saveCoarrayHandle(builder, loc, box.getAddr(), coarrayHandle);
}

/// Generate call to prif_allocate_coarray runtime subroutine
mlir::Value Fortran::lower::genAllocateCoarray(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::semantics::Symbol &sym, mlir::Type allocType) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  // Handle LCOBOUNDS and UCOBOUNDS form the Fortran::semantics::Symbol
  auto [lcobounds, ucobounds] =
      Fortran::lower::genCoarrayCoBounds(converter, loc, sym);

  // Allocate reference for allocated_memory
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Value allocMem = builder.createTemporary(loc, ptrTy);

  // TODO: Handle STAT and ERRMSG
  mlir::Value stat = builder.create<fir::AbsentOp>(
      loc, builder.getRefType(builder.getI32Type()));
  mlir::Value errMsg = builder.create<fir::AbsentOp>(
      loc, fir::BoxType::get(mlir::NoneType::get(builder.getContext())));

  mlir::Value coarrayHandle = genAllocateCoarrayRuntimeCall(
      builder, loc, allocType, lcobounds, ucobounds, allocMem, stat, errMsg);

  // Saving the coarray_handle.
  allocMem =
      builder.create<fir::LoadOp>(loc, builder.getRefType(allocType), allocMem);
  fir::runtime::saveCoarrayHandle(builder, loc, allocMem, coarrayHandle);
  return allocMem;
}

mlir::Value Fortran::lower::genDeallocateCoarray(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value coarrayAddr, mlir::Value errMsgAddr) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::FunctionType ftype = PRIF_FUNCTYPE(ptrTy, ptrTy, ptrTy, ptrTy);
  mlir::func::FuncOp funcOp =
      builder.createFunction(loc, PRIFNAME_SUB("deallocate_coarray"), ftype);

  // PRIF define prif_deallocate_coarray where the coarray_handles arg is an
  // array of handles, but this function if called by genDeallocate in
  // flang/lib/Lower/Allocatable.cpp and this function treat only one entity
  mlir::Value coarrayHandle =
      fir::runtime::getCoarrayHandle(builder, loc, coarrayAddr);
  mlir::Value stat = builder.createTemporary(loc, builder.getI32Type());
  builder.create<fir::StoreOp>(
      loc, builder.createIntegerConstant(loc, builder.getI32Type(), 0), stat);
  auto nullPtr = builder.createNullConstant(loc, ptrTy);
  llvm::SmallVector<mlir::Value> localArgs = {coarrayHandle, stat, nullPtr,
                                              errMsgAddr};
  builder.create<fir::CallOp>(loc, funcOp, localArgs);
  // TODO: Calling freeMemOp on the coarray_handle
  // freeCoarrayHandle(builder, loc, coarrayAddr);
  return builder.create<fir::LoadOp>(loc, stat);
}

//===----------------------------------------------------------------------===//
// COARRAY expressions
//===----------------------------------------------------------------------===//

fir::ExtendedValue Fortran::lower::CoarrayExprHelper::genAddr(
    const Fortran::evaluate::CoarrayRef &expr) {
  TODO(converter.getCurrentLocation(), "co-array address");
}

fir::ExtendedValue Fortran::lower::CoarrayExprHelper::genValue(
    const Fortran::evaluate::CoarrayRef &expr) {
  TODO(converter.getCurrentLocation(), "co-array value");
}
