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
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Coarray.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/tools.h"

using namespace Fortran::semantics;
using namespace Fortran::runtime;

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
    const Fortran::parser::ChangeTeamStmt &) {
  TODO(converter.getCurrentLocation(), "coarray: CHANGE TEAM statement");
}

void Fortran::lower::genEndChangeTeamStmt(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &,
    const Fortran::parser::EndChangeTeamStmt &) {
  TODO(converter.getCurrentLocation(), "coarray: END CHANGE TEAM statement");
}

void Fortran::lower::genFormTeamStatement(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &, const Fortran::parser::FormTeamStmt &) {
  TODO(converter.getCurrentLocation(), "coarray: FORM TEAM statement");
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
