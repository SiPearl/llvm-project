//===- PrifLoweringPass.h - Prif Lowering Pass ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs transformation on coarray's intrinsics/statements to call
// on PRIF interface
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_PRIFLOWERINGPASS_H
#define LLVM_TRANSFORMS_SCALAR_PRIFLOWERINGPASS_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/PassManager.h"
#include <utility>

namespace llvm {

/// Performs Prif Lowering Pass.
class PrifLoweringPass : public PassInfoMixin<PrifLoweringPass> {
public:
  PrifLoweringPass() = default;

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);

  bool
  processPrifReplacement(Module &M,
                         function_ref<TargetLibraryInfo &(Function &)> GetTLI);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_PRIFLOWERINGPASS_H
