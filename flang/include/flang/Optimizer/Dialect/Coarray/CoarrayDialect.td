//===-- CoarrayDialect.td - Coarray dialect base definitions - tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the Coarray  dialect
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DIALECT_COARRAY_COARRAYDIALECT
#define FORTRAN_DIALECT_COARRAY_COARRAYDIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def CoarrayDialect : Dialect {
  let name = "coarray";

  let summary = "Coarray dialect";

  let description = [{
    The "coarray" dialect is designed to contain the basic coarray operations
    in Fortran as descibed in the standard.
    This includes synchronization operations, atomic operations,
    image queries, teams, criticals, etc. The Coarray dialect operations use 
    the FIR types and are tightly coupled with FIR and HLFIR.
  }];

  let cppNamespace = "::coarray";
  let dependentDialects = ["fir::FIROpsDialect"];
}

#endif // FORTRAN_DIALECT_COARRAY_COARRAYDIALECT
