//===- LoopSimplifyCFG.cpp - Loop CFG Simplification Pass -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Loop SimplifyCFG Pass. This pass is responsible for
// basic loop CFG cleanup, primarily to assist other loop passes. If you
// encounter a noncanonical CFG construct that causes another loop pass to
// perform suboptimally, this is the place to fix it up.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_LOOPSIMPLIFYCFG_H
#define LLVM_TRANSFORMS_SCALAR_LOOPSIMPLIFYCFG_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class LPMUpdater;
class Loop;
class MemorySSAUpdater;

/// If L has more than one exiting block, but those all lead to the same
/// exit block, transform L so that it has a single exiting block
/// afterwards. DT, LI and SE are mandatory arguments and updated by
/// the transformation. MSSAU can be provided if it needs to be
/// updated.
bool transformLoopToSingleExit(Loop &L, DominatorTree &DT, LoopInfo &LI,
                               ScalarEvolution &SE, MemorySSAUpdater *MSSAU);

/// Performs basic CFG simplifications to assist other loop passes.
class LoopSimplifyCFGPass : public PassInfoMixin<LoopSimplifyCFGPass> {
  bool EnableToSingleExitTransform;

public:
  LoopSimplifyCFGPass(bool DoToSingleExitTransform = false)
      : EnableToSingleExitTransform(DoToSingleExitTransform) {}

  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &U);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_LOOPSIMPLIFYCFG_H
