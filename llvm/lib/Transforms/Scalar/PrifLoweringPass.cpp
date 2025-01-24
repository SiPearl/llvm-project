//===- PrifLoweringPass.cpp - PRIF lowering pass --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides the PRIF Lowering Pass
/// It consists on applying transformation for code that use Coarray features
/// in Fortran by replacing those intrinsics/statements call by the
/// corresponding PRIF subroutine
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/PrifLoweringPass.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/Transforms/Utils/Local.h"
#include <utility>

#define DEBUG_TYPE "prif-lowering"

using namespace llvm;

namespace llvm {

bool PrifLoweringPass::processPrifReplacement(
    Module &M, function_ref<TargetLibraryInfo &(Function &)> GetTLI) {
  // Finding presence of prif_* calls in the code.
  bool hasPrifCalls = false;
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *C = dyn_cast<CallInst>(&I)) {
          if (C && C->getCalledFunction() &&
              C->getCalledFunction()->getName().starts_with("_QMprifP")) {
            hasPrifCalls = true;
            break;
          }
        }
      }
    }
  }
  if (!hasPrifCalls)
    return true;

  // FIXME: Handling STAT for prif_init
  for (auto &F : M) {
    // Insert intialization of prif implementation.
    if (F.getName() == "_QQmain") {
      IRBuilder<> builder(F.getEntryBlock().getFirstNonPHI());
      auto *statArg = builder.CreateAlloca(builder.getInt32Ty(), nullptr);
      builder.CreateStore(ConstantInt::get(builder.getInt32Ty(), 0), statArg);
      builder.CreateCall(
          M.getOrInsertFunction("_QMprifPprif_init",
                                FunctionType::get(builder.getVoidTy(),
                                                  {builder.getPtrTy()}, true)),
          {statArg /*stat*/});
    }
    // Insert finalization runtime function if this is a returning basic block
    // from main (prif_stop)
    // FIXME: Using prif_stop_callback_interface as defined in PRIF 0.5 ?
    for (auto &BB : F) {
      if ((F.getName() == "_QQmain") && isa<ReturnInst>(BB.getTerminator())) {
        IRBuilder<> builder(BB.getTerminator());
        auto *codeArg = builder.CreateAlloca(builder.getInt32Ty(), nullptr);
        auto *quietArg = builder.CreateAlloca(builder.getInt1Ty(), nullptr);
        builder.CreateStore(ConstantInt::get(builder.getInt32Ty(), 0), codeArg);
        builder.CreateStore(builder.getTrue(), quietArg);
        builder.CreateCall(
            M.getOrInsertFunction(
                "_QMprifPprif_stop",
                FunctionType::get(builder.getVoidTy(),
                                  {builder.getPtrTy(), builder.getPtrTy(),
                                   builder.getPtrTy()},
                                  true)),
            {quietArg /*quiet*/, codeArg /*code_int*/,
             ConstantPointerNull::get(builder.getPtrTy()) /*code_char*/});
      }
    }
  }
  return true;
}

PreservedAnalyses PrifLoweringPass::run(Module &M, ModuleAnalysisManager &MAM) {
  LLVM_DEBUG(dbgs() << "==== PRIF Lowering Pass : START..." << "\n");

  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetTLI = [&FAM](Function &F) -> TargetLibraryInfo & {
    return FAM.getResult<TargetLibraryAnalysis>(F);
  };

  if (!processPrifReplacement(M, GetTLI)) {
    LLVM_DEBUG(llvm::dbgs() << "PRIF Replacement Failed.\n";);
    return PreservedAnalyses::all();
  }

  LLVM_DEBUG(dbgs() << "==== PRIF Lowering Pass : END!" << "\n");
  return PreservedAnalyses::all();
}

}; // namespace llvm
