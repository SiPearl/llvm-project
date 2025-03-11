//===-- VPlanTransforms.cpp - Utility VPlan to VPlan transforms -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a set of utility VPlan to VPlan transformations.
///
//===----------------------------------------------------------------------===//

#include "VPlanTransforms.h"
#include "VPRecipeBuilder.h"
#include "VPlan.h"
#include "VPlanAnalysis.h"
#include "VPlanCFG.h"
#include "VPlanDominatorTree.h"
#include "VPlanPatternMatch.h"
#include "VPlanUtils.h"
#include "VPlanVerifier.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/TypeSize.h"
#include <optional>

#define DEBUG_TYPE "vplan"

using namespace llvm;

void VPlanTransforms::VPInstructionsToVPRecipes(
    VPlanPtr &Plan,
    function_ref<const InductionDescriptor *(PHINode *)>
        GetIntOrFpInductionDescriptor,
    ScalarEvolution &SE, const TargetLibraryInfo &TLI) {

  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>> RPOT(
      Plan->getVectorLoopRegion());
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT)) {
    // Skip blocks outside region
    if (!VPBB->getParent())
      break;
    VPRecipeBase *Term = VPBB->getTerminator();
    auto EndIter = Term ? Term->getIterator() : VPBB->end();
    // Introduce each ingredient into VPlan.
    for (VPRecipeBase &Ingredient :
         make_early_inc_range(make_range(VPBB->begin(), EndIter))) {

      VPValue *VPV = Ingredient.getVPSingleValue();
      Instruction *Inst = cast<Instruction>(VPV->getUnderlyingValue());

      VPRecipeBase *NewRecipe = nullptr;
      if (auto *VPPhi = dyn_cast<VPWidenPHIRecipe>(&Ingredient)) {
        auto *Phi = cast<PHINode>(VPPhi->getUnderlyingValue());
        const auto *II = GetIntOrFpInductionDescriptor(Phi);
        if (!II)
          continue;

        VPValue *Start = Plan->getOrAddLiveIn(II->getStartValue());
        VPValue *Step =
            vputils::getOrCreateVPValueForSCEVExpr(*Plan, II->getStep(), SE);
        NewRecipe = new VPWidenIntOrFpInductionRecipe(
            Phi, Start, Step, &Plan->getVF(), *II, Ingredient.getDebugLoc());
      } else {
        assert(isa<VPInstruction>(&Ingredient) &&
               "only VPInstructions expected here");
        assert(!isa<PHINode>(Inst) && "phis should be handled above");
        // Create VPWidenMemoryRecipe for loads and stores.
        if (LoadInst *Load = dyn_cast<LoadInst>(Inst)) {
          NewRecipe = new VPWidenLoadRecipe(
              *Load, Ingredient.getOperand(0), nullptr /*Mask*/,
              false /*Consecutive*/, false /*Reverse*/,
              Ingredient.getDebugLoc());
        } else if (StoreInst *Store = dyn_cast<StoreInst>(Inst)) {
          NewRecipe = new VPWidenStoreRecipe(
              *Store, Ingredient.getOperand(1), Ingredient.getOperand(0),
              nullptr /*Mask*/, false /*Consecutive*/, false /*Reverse*/,
              Ingredient.getDebugLoc());
        } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Inst)) {
          NewRecipe = new VPWidenGEPRecipe(GEP, Ingredient.operands());
        } else if (CallInst *CI = dyn_cast<CallInst>(Inst)) {
          NewRecipe = new VPWidenIntrinsicRecipe(
              *CI, getVectorIntrinsicIDForCall(CI, &TLI),
              {Ingredient.op_begin(), Ingredient.op_end() - 1}, CI->getType(),
              CI->getDebugLoc());
        } else if (SelectInst *SI = dyn_cast<SelectInst>(Inst)) {
          NewRecipe = new VPWidenSelectRecipe(*SI, Ingredient.operands());
        } else if (auto *CI = dyn_cast<CastInst>(Inst)) {
          NewRecipe = new VPWidenCastRecipe(
              CI->getOpcode(), Ingredient.getOperand(0), CI->getType(), *CI);
        } else {
          NewRecipe = new VPWidenRecipe(*Inst, Ingredient.operands());
        }
      }

      NewRecipe->insertBefore(&Ingredient);
      if (NewRecipe->getNumDefinedValues() == 1)
        VPV->replaceAllUsesWith(NewRecipe->getVPSingleValue());
      else
        assert(NewRecipe->getNumDefinedValues() == 0 &&
               "Only recpies with zero or one defined values expected");
      Ingredient.eraseFromParent();
    }
  }
}

static bool sinkScalarOperands(VPlan &Plan) {
  auto Iter = vp_depth_first_deep(Plan.getEntry());
  bool Changed = false;
  // First, collect the operands of all recipes in replicate blocks as seeds for
  // sinking.
  SetVector<std::pair<VPBasicBlock *, VPSingleDefRecipe *>> WorkList;
  for (VPRegionBlock *VPR : VPBlockUtils::blocksOnly<VPRegionBlock>(Iter)) {
    VPBasicBlock *EntryVPBB = VPR->getEntryBasicBlock();
    if (!VPR->isReplicator() || EntryVPBB->getSuccessors().size() != 2)
      continue;
    VPBasicBlock *VPBB = dyn_cast<VPBasicBlock>(EntryVPBB->getSuccessors()[0]);
    if (!VPBB || VPBB->getSingleSuccessor() != VPR->getExitingBasicBlock())
      continue;
    for (auto &Recipe : *VPBB) {
      for (VPValue *Op : Recipe.operands())
        if (auto *Def =
                dyn_cast_or_null<VPSingleDefRecipe>(Op->getDefiningRecipe()))
          WorkList.insert(std::make_pair(VPBB, Def));
    }
  }

  bool ScalarVFOnly = Plan.hasScalarVFOnly();
  // Try to sink each replicate or scalar IV steps recipe in the worklist.
  for (unsigned I = 0; I != WorkList.size(); ++I) {
    VPBasicBlock *SinkTo;
    VPSingleDefRecipe *SinkCandidate;
    std::tie(SinkTo, SinkCandidate) = WorkList[I];
    if (SinkCandidate->getParent() == SinkTo ||
        SinkCandidate->mayHaveSideEffects() ||
        SinkCandidate->mayReadOrWriteMemory())
      continue;
    if (auto *RepR = dyn_cast<VPReplicateRecipe>(SinkCandidate)) {
      if (!ScalarVFOnly && RepR->isUniform())
        continue;
    } else if (!isa<VPScalarIVStepsRecipe>(SinkCandidate))
      continue;

    bool NeedsDuplicating = false;
    // All recipe users of the sink candidate must be in the same block SinkTo
    // or all users outside of SinkTo must be uniform-after-vectorization (
    // i.e., only first lane is used) . In the latter case, we need to duplicate
    // SinkCandidate.
    auto CanSinkWithUser = [SinkTo, &NeedsDuplicating,
                            SinkCandidate](VPUser *U) {
      auto *UI = cast<VPRecipeBase>(U);
      if (UI->getParent() == SinkTo)
        return true;
      NeedsDuplicating = UI->onlyFirstLaneUsed(SinkCandidate);
      // We only know how to duplicate VPRecipeRecipes for now.
      return NeedsDuplicating && isa<VPReplicateRecipe>(SinkCandidate);
    };
    if (!all_of(SinkCandidate->users(), CanSinkWithUser))
      continue;

    if (NeedsDuplicating) {
      if (ScalarVFOnly)
        continue;
      Instruction *I = SinkCandidate->getUnderlyingInstr();
      auto *Clone = new VPReplicateRecipe(I, SinkCandidate->operands(), true);
      // TODO: add ".cloned" suffix to name of Clone's VPValue.

      Clone->insertBefore(SinkCandidate);
      SinkCandidate->replaceUsesWithIf(Clone, [SinkTo](VPUser &U, unsigned) {
        return cast<VPRecipeBase>(&U)->getParent() != SinkTo;
      });
    }
    SinkCandidate->moveBefore(*SinkTo, SinkTo->getFirstNonPhi());
    for (VPValue *Op : SinkCandidate->operands())
      if (auto *Def =
              dyn_cast_or_null<VPSingleDefRecipe>(Op->getDefiningRecipe()))
        WorkList.insert(std::make_pair(SinkTo, Def));
    Changed = true;
  }
  return Changed;
}

/// If \p R is a region with a VPBranchOnMaskRecipe in the entry block, return
/// the mask.
VPValue *getPredicatedMask(VPRegionBlock *R) {
  auto *EntryBB = dyn_cast<VPBasicBlock>(R->getEntry());
  if (!EntryBB || EntryBB->size() != 1 ||
      !isa<VPBranchOnMaskRecipe>(EntryBB->begin()))
    return nullptr;

  return cast<VPBranchOnMaskRecipe>(&*EntryBB->begin())->getOperand(0);
}

/// If \p R is a triangle region, return the 'then' block of the triangle.
static VPBasicBlock *getPredicatedThenBlock(VPRegionBlock *R) {
  auto *EntryBB = cast<VPBasicBlock>(R->getEntry());
  if (EntryBB->getNumSuccessors() != 2)
    return nullptr;

  auto *Succ0 = dyn_cast<VPBasicBlock>(EntryBB->getSuccessors()[0]);
  auto *Succ1 = dyn_cast<VPBasicBlock>(EntryBB->getSuccessors()[1]);
  if (!Succ0 || !Succ1)
    return nullptr;

  if (Succ0->getNumSuccessors() + Succ1->getNumSuccessors() != 1)
    return nullptr;
  if (Succ0->getSingleSuccessor() == Succ1)
    return Succ0;
  if (Succ1->getSingleSuccessor() == Succ0)
    return Succ1;
  return nullptr;
}

// Merge replicate regions in their successor region, if a replicate region
// is connected to a successor replicate region with the same predicate by a
// single, empty VPBasicBlock.
static bool mergeReplicateRegionsIntoSuccessors(VPlan &Plan) {
  SmallPtrSet<VPRegionBlock *, 4> TransformedRegions;

  // Collect replicate regions followed by an empty block, followed by another
  // replicate region with matching masks to process front. This is to avoid
  // iterator invalidation issues while merging regions.
  SmallVector<VPRegionBlock *, 8> WorkList;
  for (VPRegionBlock *Region1 : VPBlockUtils::blocksOnly<VPRegionBlock>(
           vp_depth_first_deep(Plan.getEntry()))) {
    if (!Region1->isReplicator())
      continue;
    auto *MiddleBasicBlock =
        dyn_cast_or_null<VPBasicBlock>(Region1->getSingleSuccessor());
    if (!MiddleBasicBlock || !MiddleBasicBlock->empty())
      continue;

    auto *Region2 =
        dyn_cast_or_null<VPRegionBlock>(MiddleBasicBlock->getSingleSuccessor());
    if (!Region2 || !Region2->isReplicator())
      continue;

    VPValue *Mask1 = getPredicatedMask(Region1);
    VPValue *Mask2 = getPredicatedMask(Region2);
    if (!Mask1 || Mask1 != Mask2)
      continue;

    assert(Mask1 && Mask2 && "both region must have conditions");
    WorkList.push_back(Region1);
  }

  // Move recipes from Region1 to its successor region, if both are triangles.
  for (VPRegionBlock *Region1 : WorkList) {
    if (TransformedRegions.contains(Region1))
      continue;
    auto *MiddleBasicBlock = cast<VPBasicBlock>(Region1->getSingleSuccessor());
    auto *Region2 = cast<VPRegionBlock>(MiddleBasicBlock->getSingleSuccessor());

    VPBasicBlock *Then1 = getPredicatedThenBlock(Region1);
    VPBasicBlock *Then2 = getPredicatedThenBlock(Region2);
    if (!Then1 || !Then2)
      continue;

    // Note: No fusion-preventing memory dependencies are expected in either
    // region. Such dependencies should be rejected during earlier dependence
    // checks, which guarantee accesses can be re-ordered for vectorization.
    //
    // Move recipes to the successor region.
    for (VPRecipeBase &ToMove : make_early_inc_range(reverse(*Then1)))
      ToMove.moveBefore(*Then2, Then2->getFirstNonPhi());

    auto *Merge1 = cast<VPBasicBlock>(Then1->getSingleSuccessor());
    auto *Merge2 = cast<VPBasicBlock>(Then2->getSingleSuccessor());

    // Move VPPredInstPHIRecipes from the merge block to the successor region's
    // merge block. Update all users inside the successor region to use the
    // original values.
    for (VPRecipeBase &Phi1ToMove : make_early_inc_range(reverse(*Merge1))) {
      VPValue *PredInst1 =
          cast<VPPredInstPHIRecipe>(&Phi1ToMove)->getOperand(0);
      VPValue *Phi1ToMoveV = Phi1ToMove.getVPSingleValue();
      Phi1ToMoveV->replaceUsesWithIf(PredInst1, [Then2](VPUser &U, unsigned) {
        return cast<VPRecipeBase>(&U)->getParent() == Then2;
      });

      // Remove phi recipes that are unused after merging the regions.
      if (Phi1ToMove.getVPSingleValue()->getNumUsers() == 0) {
        Phi1ToMove.eraseFromParent();
        continue;
      }
      Phi1ToMove.moveBefore(*Merge2, Merge2->begin());
    }

    // Remove the dead recipes in Region1's entry block.
    for (VPRecipeBase &R :
         make_early_inc_range(reverse(*Region1->getEntryBasicBlock())))
      R.eraseFromParent();

    // Finally, remove the first region.
    for (VPBlockBase *Pred : make_early_inc_range(Region1->getPredecessors())) {
      VPBlockUtils::disconnectBlocks(Pred, Region1);
      VPBlockUtils::connectBlocks(Pred, MiddleBasicBlock);
    }
    VPBlockUtils::disconnectBlocks(Region1, MiddleBasicBlock);
    TransformedRegions.insert(Region1);
  }

  return !TransformedRegions.empty();
}

static VPRegionBlock *createReplicateRegion(VPReplicateRecipe *PredRecipe,
                                            VPlan &Plan) {
  Instruction *Instr = PredRecipe->getUnderlyingInstr();
  // Build the triangular if-then region.
  std::string RegionName = (Twine("pred.") + Instr->getOpcodeName()).str();
  assert(Instr->getParent() && "Predicated instruction not in any basic block");
  auto *BlockInMask = PredRecipe->getMask();
  auto *MaskDef = BlockInMask->getDefiningRecipe();
  auto *BOMRecipe = new VPBranchOnMaskRecipe(
      BlockInMask, MaskDef ? MaskDef->getDebugLoc() : DebugLoc());
  auto *Entry =
      Plan.createVPBasicBlock(Twine(RegionName) + ".entry", BOMRecipe);

  // Replace predicated replicate recipe with a replicate recipe without a
  // mask but in the replicate region.
  auto *RecipeWithoutMask = new VPReplicateRecipe(
      PredRecipe->getUnderlyingInstr(),
      make_range(PredRecipe->op_begin(), std::prev(PredRecipe->op_end())),
      PredRecipe->isUniform());
  auto *Pred =
      Plan.createVPBasicBlock(Twine(RegionName) + ".if", RecipeWithoutMask);

  VPPredInstPHIRecipe *PHIRecipe = nullptr;
  if (PredRecipe->getNumUsers() != 0) {
    PHIRecipe = new VPPredInstPHIRecipe(RecipeWithoutMask,
                                        RecipeWithoutMask->getDebugLoc());
    PredRecipe->replaceAllUsesWith(PHIRecipe);
    PHIRecipe->setOperand(0, RecipeWithoutMask);
  }
  PredRecipe->eraseFromParent();
  auto *Exiting =
      Plan.createVPBasicBlock(Twine(RegionName) + ".continue", PHIRecipe);
  VPRegionBlock *Region =
      Plan.createVPRegionBlock(Entry, Exiting, RegionName, true);

  // Note: first set Entry as region entry and then connect successors starting
  // from it in order, to propagate the "parent" of each VPBasicBlock.
  VPBlockUtils::insertTwoBlocksAfter(Pred, Exiting, Entry);
  VPBlockUtils::connectBlocks(Pred, Exiting);

  return Region;
}

static void addReplicateRegions(VPlan &Plan) {
  SmallVector<VPReplicateRecipe *> WorkList;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getEntry()))) {
    for (VPRecipeBase &R : *VPBB)
      if (auto *RepR = dyn_cast<VPReplicateRecipe>(&R)) {
        if (RepR->isPredicated())
          WorkList.push_back(RepR);
      }
  }

  unsigned BBNum = 0;
  for (VPReplicateRecipe *RepR : WorkList) {
    VPBasicBlock *CurrentBlock = RepR->getParent();
    VPBasicBlock *SplitBlock = CurrentBlock->splitAt(RepR->getIterator());

    BasicBlock *OrigBB = RepR->getUnderlyingInstr()->getParent();
    SplitBlock->setName(
        OrigBB->hasName() ? OrigBB->getName() + "." + Twine(BBNum++) : "");
    // Record predicated instructions for above packing optimizations.
    VPBlockBase *Region = createReplicateRegion(RepR, Plan);
    Region->setParent(CurrentBlock->getParent());
    VPBlockUtils::insertOnEdge(CurrentBlock, SplitBlock, Region);
  }
}

/// Remove redundant VPBasicBlocks by merging them into their predecessor if
/// the predecessor has a single successor.
static bool mergeBlocksIntoPredecessors(VPlan &Plan) {
  SmallVector<VPBasicBlock *> WorkList;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getEntry()))) {
    // Don't fold the blocks in the skeleton of the Plan into their single
    // predecessors for now.
    // TODO: Remove restriction once more of the skeleton is modeled in VPlan.
    if (!VPBB->getParent())
      continue;
    auto *PredVPBB =
        dyn_cast_or_null<VPBasicBlock>(VPBB->getSinglePredecessor());
    if (!PredVPBB || PredVPBB->getNumSuccessors() != 1 ||
        isa<VPIRBasicBlock>(PredVPBB))
      continue;
    WorkList.push_back(VPBB);
  }

  for (VPBasicBlock *VPBB : WorkList) {
    VPBasicBlock *PredVPBB = cast<VPBasicBlock>(VPBB->getSinglePredecessor());
    for (VPRecipeBase &R : make_early_inc_range(*VPBB))
      R.moveBefore(*PredVPBB, PredVPBB->end());
    VPBlockUtils::disconnectBlocks(PredVPBB, VPBB);
    auto *ParentRegion = VPBB->getParent();
    if (ParentRegion && ParentRegion->getExiting() == VPBB)
      ParentRegion->setExiting(PredVPBB);
    for (auto *Succ : to_vector(VPBB->successors()))
      VPBlockUtils::replacePredecessor(VPBB, PredVPBB, Succ);

    // VPBB is now dead and will be cleaned up when the plan gets destroyed.
  }
  return !WorkList.empty();
}

void VPlanTransforms::createAndOptimizeReplicateRegions(VPlan &Plan) {
  // Convert masked VPReplicateRecipes to if-then region blocks.
  addReplicateRegions(Plan);

  bool ShouldSimplify = true;
  while (ShouldSimplify) {
    ShouldSimplify = sinkScalarOperands(Plan);
    ShouldSimplify |= mergeReplicateRegionsIntoSuccessors(Plan);
    ShouldSimplify |= mergeBlocksIntoPredecessors(Plan);
  }
}

// Return true if the mask argument is known to always have at least
// one lane active.
static bool maskKnownToHaveActiveLane(VPValue *V) {
  if (auto *Phi = dyn_cast<VPWidenPHIRecipe>(V))
    return all_of(Phi->operands(), maskKnownToHaveActiveLane) ||
           (Phi->isActiveLaneMask() &&
            maskKnownToHaveActiveLane(Phi->getOperand(0)));

  using namespace VPlanPatternMatch;
  return match(V, m_True()) || isa<VPActiveLaneMaskPHIRecipe>(V);
}

void VPlanTransforms::handleMaskedUniformReplicateRecipes(VPlan &Plan) {
  // Find any predicated uniform replication recipes.
  SmallVector<VPReplicateRecipe *> WorkList;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getVectorLoopRegion())))
    if (auto *Region = VPBB->getParent(); Region && !Region->isReplicator())
      for (VPRecipeBase &R : *VPBB)
        if (auto *Rep = dyn_cast<VPReplicateRecipe>(&R);
            Rep && Rep->isUniform() && Rep->isPredicated() &&
            Rep->mayReadOrWriteMemory())
          WorkList.push_back(Rep);

  // Build a list of recipes (in reverse topological order) that can be
  // sunken into the same basic block as Seed.
  auto BuildSinkableList =
      [&](VPReplicateRecipe *Root, VPBasicBlock *OrigBB,
          VPBasicBlock *AnyActiveBB) -> SmallSetVector<VPRecipeBase *, 4> {
    SmallSetVector<VPRecipeBase *, 4> ToSink;
    SmallVector<VPValue *, 4> WorkList;
    WorkList.append(Root->op_begin(), Root->op_end());
    while (!WorkList.empty()) {
      VPValue *V = WorkList.pop_back_val();
      VPRecipeBase *Def = V->getDefiningRecipe();
      // Don't sink side-effecting recipes or ones from
      // different basic blocks.
      if (!Def || ToSink.contains(Def) || Def->getParent() != OrigBB ||
          Def->isPhi() || Def->mayHaveSideEffects() ||
          Def->mayReadOrWriteMemory())
        continue;

      // Don't sink if there is any user not also beeing sunken.
      // This also ensures a topological order of the sunken recipes.
      if (any_of(Def->definedValues(), [&](VPValue *V) -> bool {
            return any_of(V->users(), [&](VPUser *U) -> bool {
              auto *UR = cast<VPRecipeBase>(U);
              return UR->getParent() != Root->getParent() &&
                     !ToSink.contains(UR);
            });
          }))
        continue;

      ToSink.insert(Def);
      WorkList.append(Def->op_begin(), Def->op_end());
    }

    return ToSink;
  };

  VPValue *PrevMask = nullptr;
  VPBasicBlock *PrevBB = nullptr;

  VPTypeAnalysis TypeInfo(Plan.getCanonicalIV()->getScalarType());
  for (VPReplicateRecipe *R : reverse(WorkList)) {
    VPValue *Mask = R->getMask();
    // If the mask is known to always have at least one active lane,
    // the mask can just be dropped. Otherwise, a check at runtime is needed
    // that tests if any lane is active.
    if (!maskKnownToHaveActiveLane(Mask)) {
      VPBasicBlock *Pred = R->getParent(), *IfAny = nullptr, *Succ = nullptr;
      // If at the end of the same basic block, there already is a branch around
      // a uniform memory access, and the mask is the same, and there are no
      // side-effecting instructions between this recipe and that block, then
      // reuse the existing branch.
      if (Mask == PrevMask && Pred == PrevBB &&
          std::all_of(
              ++R->getIterator(), PrevBB->getTerminator()->getIterator(),
              [&](VPRecipeBase &R) {
                return !R.mayHaveSideEffects() && !R.mayReadOrWriteMemory();
              })) {
        IfAny = cast<VPBasicBlock>(Pred->getSuccessors()[0]);
        Succ = cast<VPBasicBlock>(Pred->getSuccessors()[1]);
        R->moveBefore(*IfAny, IfAny->begin());
        assert(IfAny->getSingleSuccessor() == Succ);
      } else {
        IfAny = Pred->splitAt(R->getIterator());
        Succ = IfAny->splitAt(++R->getIterator());

        IfAny->setName(Pred->getName() + ".anyactive");
        Succ->setName(Pred->getName() + ".join");
        auto *AnyOf = new VPInstruction(VPInstruction::AnyOf, {Mask});
        AnyOf->insertBefore(*Pred, Pred->end());
        auto *CondBr = new VPInstruction(VPInstruction::BranchOnCond, {AnyOf});
        CondBr->insertBefore(*Pred, Pred->end());
        VPBlockUtils::connectBlocks(Pred, Succ);

        PrevMask = Mask;
        PrevBB = Pred;
      }

      // TODO: Use a scalar phi as soon as those are available.
      auto *Phi = new VPWidenPHIRecipe(nullptr);
      R->replaceAllUsesWith(Phi);
      Phi->insertBefore(*Succ, Succ->begin());
      Phi->addOperand(R);
      Phi->addOperand(
          Plan.getOrAddLiveIn(PoisonValue::get(TypeInfo.inferScalarType(R))));

      // Sink other instructions only used by this one inside the IfAny block.
      auto ToSink = BuildSinkableList(R, Pred, IfAny);
      for (VPRecipeBase *SinkMe : reverse(ToSink))
        SinkMe->moveBefore(*IfAny, R->getIterator());
    }

    auto *NewR = new VPReplicateRecipe(R->getUnderlyingInstr(),
                                       drop_end(R->operands()), true, nullptr);
    NewR->insertBefore(R);
    R->replaceAllUsesWith(NewR);
    R->eraseFromParent();
  }
}

/// Remove redundant casts of inductions.
///
/// Such redundant casts are casts of induction variables that can be ignored,
/// because we already proved that the casted phi is equal to the uncasted phi
/// in the vectorized loop. There is no need to vectorize the cast - the same
/// value can be used for both the phi and casts in the vector loop.
static void removeRedundantInductionCasts(VPlan &Plan) {
  for (auto &Phi : Plan.getVectorLoopRegion()->getEntryBasicBlock()->phis()) {
    auto *IV = dyn_cast<VPWidenIntOrFpInductionRecipe>(&Phi);
    if (!IV || IV->getTruncInst())
      continue;

    // A sequence of IR Casts has potentially been recorded for IV, which
    // *must be bypassed* when the IV is vectorized, because the vectorized IV
    // will produce the desired casted value. This sequence forms a def-use
    // chain and is provided in reverse order, ending with the cast that uses
    // the IV phi. Search for the recipe of the last cast in the chain and
    // replace it with the original IV. Note that only the final cast is
    // expected to have users outside the cast-chain and the dead casts left
    // over will be cleaned up later.
    auto &Casts = IV->getInductionDescriptor().getCastInsts();
    VPValue *FindMyCast = IV;
    for (Instruction *IRCast : reverse(Casts)) {
      VPSingleDefRecipe *FoundUserCast = nullptr;
      for (auto *U : FindMyCast->users()) {
        auto *UserCast = dyn_cast<VPSingleDefRecipe>(U);
        if (UserCast && UserCast->getUnderlyingValue() == IRCast) {
          FoundUserCast = UserCast;
          break;
        }
      }
      FindMyCast = FoundUserCast;
    }
    FindMyCast->replaceAllUsesWith(IV);
  }
}

/// Try to replace VPWidenCanonicalIVRecipes with a widened canonical IV
/// recipe, if it exists.
static void removeRedundantCanonicalIVs(VPlan &Plan) {
  VPCanonicalIVPHIRecipe *CanonicalIV = Plan.getCanonicalIV();
  VPWidenCanonicalIVRecipe *WidenNewIV = nullptr;
  for (VPUser *U : CanonicalIV->users()) {
    WidenNewIV = dyn_cast<VPWidenCanonicalIVRecipe>(U);
    if (WidenNewIV)
      break;
  }

  if (!WidenNewIV)
    return;

  VPBasicBlock *HeaderVPBB = Plan.getVectorLoopRegion()->getEntryBasicBlock();
  for (VPRecipeBase &Phi : HeaderVPBB->phis()) {
    auto *WidenOriginalIV = dyn_cast<VPWidenIntOrFpInductionRecipe>(&Phi);

    if (!WidenOriginalIV || !WidenOriginalIV->isCanonical())
      continue;

    // Replace WidenNewIV with WidenOriginalIV if WidenOriginalIV provides
    // everything WidenNewIV's users need. That is, WidenOriginalIV will
    // generate a vector phi or all users of WidenNewIV demand the first lane
    // only.
    if (any_of(WidenOriginalIV->users(),
               [WidenOriginalIV](VPUser *U) {
                 return !U->usesScalars(WidenOriginalIV);
               }) ||
        vputils::onlyFirstLaneUsed(WidenNewIV)) {
      WidenNewIV->replaceAllUsesWith(WidenOriginalIV);
      WidenNewIV->eraseFromParent();
      return;
    }
  }
}

/// Returns true if \p R is dead and can be removed.
static bool isDeadRecipe(VPRecipeBase &R) {
  using namespace llvm::PatternMatch;
  // Do remove conditional assume instructions as their conditions may be
  // flattened.
  auto *RepR = dyn_cast<VPReplicateRecipe>(&R);
  bool IsConditionalAssume =
      RepR && RepR->isPredicated() &&
      match(RepR->getUnderlyingInstr(), m_Intrinsic<Intrinsic::assume>());
  if (IsConditionalAssume)
    return true;

  if (R.mayHaveSideEffects())
    return false;

  // Recipe is dead if no user keeps the recipe alive.
  return all_of(R.definedValues(),
                [](VPValue *V) { return V->getNumUsers() == 0; });
}

void VPlanTransforms::removeDeadRecipes(VPlan &Plan) {
  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>> RPOT(
      Plan.getEntry());

  for (VPBasicBlock *VPBB : reverse(VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT))) {
    // The recipes in the block are processed in reverse order, to catch chains
    // of dead recipes.
    for (VPRecipeBase &R : make_early_inc_range(reverse(*VPBB))) {
      if (isDeadRecipe(R))
        R.eraseFromParent();
    }
  }
}

static VPScalarIVStepsRecipe *
createScalarIVSteps(VPlan &Plan, InductionDescriptor::InductionKind Kind,
                    Instruction::BinaryOps InductionOpcode,
                    FPMathOperator *FPBinOp, Instruction *TruncI,
                    VPValue *StartV, VPValue *Step, DebugLoc DL,
                    VPBuilder &Builder) {
  VPBasicBlock *HeaderVPBB = Plan.getVectorLoopRegion()->getEntryBasicBlock();
  VPCanonicalIVPHIRecipe *CanonicalIV = Plan.getCanonicalIV();
  VPSingleDefRecipe *BaseIV = Builder.createDerivedIV(
      Kind, FPBinOp, StartV, CanonicalIV, Step, "offset.idx");

  // Truncate base induction if needed.
  Type *CanonicalIVType = CanonicalIV->getScalarType();
  VPTypeAnalysis TypeInfo(CanonicalIVType);
  Type *ResultTy = TypeInfo.inferScalarType(BaseIV);
  if (TruncI) {
    Type *TruncTy = TruncI->getType();
    assert(ResultTy->getScalarSizeInBits() > TruncTy->getScalarSizeInBits() &&
           "Not truncating.");
    assert(ResultTy->isIntegerTy() && "Truncation requires an integer type");
    BaseIV = Builder.createScalarCast(Instruction::Trunc, BaseIV, TruncTy, DL);
    ResultTy = TruncTy;
  }

  // Truncate step if needed.
  Type *StepTy = TypeInfo.inferScalarType(Step);
  if (ResultTy != StepTy) {
    assert(StepTy->getScalarSizeInBits() > ResultTy->getScalarSizeInBits() &&
           "Not truncating.");
    assert(StepTy->isIntegerTy() && "Truncation requires an integer type");
    auto *VecPreheader =
        cast<VPBasicBlock>(HeaderVPBB->getSingleHierarchicalPredecessor());
    VPBuilder::InsertPointGuard Guard(Builder);
    Builder.setInsertPoint(VecPreheader);
    Step = Builder.createScalarCast(Instruction::Trunc, Step, ResultTy, DL);
  }
  return Builder.createScalarIVSteps(InductionOpcode, FPBinOp, BaseIV, Step);
}

// Create VPScalarPHIRecipe phis instead of widened ones for inner-loop
// inductions with uses of the first lane only. This avoids unnecessary
// extractelement operations.
static void optimizeInnerLoopInductions(VPlan &Plan) {
  auto HandleHeaderPhi = [](VPWidenPHIRecipe &Phi, VPRegionBlock &Region) {
    VPValue *Start = Phi.getIncomingValueForBlock(
        Region.getSinglePredecessor()->getExitingBasicBlock());
    VPValue *Next = Phi.getIncomingValueForBlock(Region.getExitingBasicBlock());

    // The final value (influenced by the loop trip-count) does not matter for
    // this purpose, but start and step need to be uniform.
    if (!vputils::isUniformAfterVectorization(Start) ||
        !vputils::isUniformAfterVectorization(Next))
      return;

    SmallVector<VPRecipeBase *> FirstLaneUsers;
    for (VPUser *U : Phi.users())
      if (auto *R = dyn_cast<VPRecipeBase>(U); R->onlyFirstLaneUsed(&Phi))
        FirstLaneUsers.push_back(R);
    if (FirstLaneUsers.empty())
      return;

    auto *IRPhi = Phi.getUnderlyingValue();
    auto *NewPhi =
        new VPScalarPHIRecipe(Start, Next, Phi.getDebugLoc(), IRPhi->getName());
    NewPhi->setUnderlyingValue(IRPhi);
    NewPhi->insertAfter(&Phi);
    for (VPRecipeBase *R : FirstLaneUsers)
      for (unsigned I = 0, N = R->getNumOperands(); I < N; ++I)
        if (R->getOperand(I) == &Phi)
          R->setOperand(I, NewPhi);

    if (Phi.getNumUsers() == 0)
      Phi.eraseFromParent();
  };

  auto HandleLCSSAPhi = [](VPWidenPHIRecipe &LCSSAPhi, VPRegionBlock &Region) {
    assert(LCSSAPhi.getNumOperands() == 1);
    auto *Sel = dyn_cast<VPInstruction>(LCSSAPhi.getOperand(0));
    if (!Sel || Sel->getOpcode() != Instruction::Select ||
        Sel->getParent()->getParent() != &Region)
      return;

    auto *ALM = dyn_cast<VPWidenPHIRecipe>(Sel->getOperand(0));
    auto *OrigOutVal = Sel->getOperand(1);
    auto *PassthroughPhi = dyn_cast<VPWidenPHIRecipe>(Sel->getOperand(2));
    if (!ALM || !ALM->isActiveLaneMask() || !PassthroughPhi ||
        PassthroughPhi->getOperand(1) != Sel)
      return;

    for (VPUser *U : OrigOutVal->users())
      if (auto *Phi = dyn_cast<VPWidenPHIRecipe>(U);
          Phi && Phi->getParent() == Region.getEntry()) {
        Sel->setOperand(2, Phi);
        Phi->setOperand(1, Sel);
        PassthroughPhi->eraseFromParent();
      }
  };

  // Traverse the phis in entry blocks of inner-loop regions.
  auto Iter = vp_depth_first_deep(Plan.getEntry());
  for (VPRegionBlock *VPR : VPBlockUtils::blocksOnly<VPRegionBlock>(Iter))
    if (!VPR->isReplicator() && VPR != Plan.getVectorLoopRegion()) {
      for (VPRecipeBase &R :
           make_early_inc_range(VPR->getEntryBasicBlock()->phis()))
        if (auto *Phi = dyn_cast<VPWidenPHIRecipe>(&R))
          HandleHeaderPhi(*Phi, *VPR);

      // Try to fuse LCSSA live-out passthrough phis with existing
      // other header phis.
      for (VPRecipeBase &R :
           cast<VPBasicBlock>(VPR->getSingleSuccessor())->phis())
        if (auto *Phi = dyn_cast<VPWidenPHIRecipe>(&R))
          HandleLCSSAPhi(*Phi, *VPR);
    }
}

static SmallVector<VPUser *> collectUsersRecursively(VPValue *V) {
  SetVector<VPUser *> Users(V->user_begin(), V->user_end());
  for (unsigned I = 0; I != Users.size(); ++I) {
    VPRecipeBase *Cur = cast<VPRecipeBase>(Users[I]);
    if (isa<VPHeaderPHIRecipe>(Cur))
      continue;
    for (VPValue *V : Cur->definedValues())
      Users.insert(V->user_begin(), V->user_end());
  }
  return Users.takeVector();
}

/// Legalize VPWidenPointerInductionRecipe, by replacing it with a PtrAdd
/// (IndStart, ScalarIVSteps (0, Step)) if only its scalar values are used, as
/// VPWidenPointerInductionRecipe will generate vectors only. If some users
/// require vectors while other require scalars, the scalar uses need to extract
/// the scalars from the generated vectors (Note that this is different to how
/// int/fp inductions are handled). Legalize extract-from-ends using uniform
/// VPReplicateRecipe of wide inductions to use regular VPReplicateRecipe, so
/// the correct end value is available. Also optimize
/// VPWidenIntOrFpInductionRecipe, if any of its users needs scalar values, by
/// providing them scalar steps built on the canonical scalar IV and update the
/// original IV's users. This is an optional optimization to reduce the needs of
/// vector extracts.
static void legalizeAndOptimizeInductions(VPlan &Plan) {
  using namespace llvm::VPlanPatternMatch;
  VPBasicBlock *HeaderVPBB = Plan.getVectorLoopRegion()->getEntryBasicBlock();
  bool HasOnlyVectorVFs = !Plan.hasScalarVFOnly();
  VPBuilder Builder(HeaderVPBB, HeaderVPBB->getFirstNonPhi());
  for (VPRecipeBase &Phi : HeaderVPBB->phis()) {
    auto *PhiR = dyn_cast<VPWidenInductionRecipe>(&Phi);
    if (!PhiR)
      continue;

    // Try to narrow wide and replicating recipes to uniform recipes, based on
    // VPlan analysis.
    // TODO: Apply to all recipes in the future, to replace legacy uniformity
    // analysis.
    auto Users = collectUsersRecursively(PhiR);
    for (VPUser *U : reverse(Users)) {
      auto *Def = dyn_cast<VPSingleDefRecipe>(U);
      auto *RepR = dyn_cast<VPReplicateRecipe>(U);
      // Skip recipes that shouldn't be narrowed.
      if (!Def || !isa<VPReplicateRecipe, VPWidenRecipe>(Def) ||
          Def->getNumUsers() == 0 || !Def->getUnderlyingValue() ||
          (RepR && (RepR->isUniform() || RepR->isPredicated())))
        continue;

      // Skip recipes that may have other lanes than their first used.
      if (!vputils::isUniformAfterVectorization(Def) &&
          !vputils::onlyFirstLaneUsed(Def))
        continue;

      auto *Clone = new VPReplicateRecipe(Def->getUnderlyingInstr(),
                                          Def->operands(), /*IsUniform*/ true);
      Clone->insertAfter(Def);
      Def->replaceAllUsesWith(Clone);
    }

    // Replace wide pointer inductions which have only their scalars used by
    // PtrAdd(IndStart, ScalarIVSteps (0, Step)).
    if (auto *PtrIV = dyn_cast<VPWidenPointerInductionRecipe>(&Phi)) {
      if (!PtrIV->onlyScalarsGenerated(Plan.hasScalableVF()))
        continue;

      const InductionDescriptor &ID = PtrIV->getInductionDescriptor();
      VPValue *StartV =
          Plan.getOrAddLiveIn(ConstantInt::get(ID.getStep()->getType(), 0));
      VPValue *StepV = PtrIV->getOperand(1);
      VPScalarIVStepsRecipe *Steps = createScalarIVSteps(
          Plan, InductionDescriptor::IK_IntInduction, Instruction::Add, nullptr,
          nullptr, StartV, StepV, PtrIV->getDebugLoc(), Builder);

      VPValue *PtrAdd = Builder.createPtrAdd(PtrIV->getStartValue(), Steps,
                                             PtrIV->getDebugLoc(), "next.gep");

      PtrIV->replaceAllUsesWith(PtrAdd);
      continue;
    }

    // Replace widened induction with scalar steps for users that only use
    // scalars.
    auto *WideIV = cast<VPWidenIntOrFpInductionRecipe>(&Phi);
    if (HasOnlyVectorVFs && none_of(WideIV->users(), [WideIV](VPUser *U) {
          return U->usesScalars(WideIV);
        }))
      continue;

    const InductionDescriptor &ID = WideIV->getInductionDescriptor();
    VPScalarIVStepsRecipe *Steps = createScalarIVSteps(
        Plan, ID.getKind(), ID.getInductionOpcode(),
        dyn_cast_or_null<FPMathOperator>(ID.getInductionBinOp()),
        WideIV->getTruncInst(), WideIV->getStartValue(), WideIV->getStepValue(),
        WideIV->getDebugLoc(), Builder);

    // Update scalar users of IV to use Step instead.
    if (!HasOnlyVectorVFs)
      WideIV->replaceAllUsesWith(Steps);
    else
      WideIV->replaceUsesWithIf(Steps, [WideIV](VPUser &U, unsigned) {
        return U.usesScalars(WideIV);
      });
  }
}

/// Check if \p VPV is an untruncated wide induction, either before or after the
/// increment. If so return the header IV (before the increment), otherwise
/// return null.
static VPWidenInductionRecipe *getOptimizableIVOf(VPValue *VPV) {
  auto *WideIV = dyn_cast<VPWidenInductionRecipe>(VPV);
  if (WideIV) {
    // VPV itself is a wide induction, separately compute the end value for exit
    // users if it is not a truncated IV.
    auto *IntOrFpIV = dyn_cast<VPWidenIntOrFpInductionRecipe>(WideIV);
    return (IntOrFpIV && IntOrFpIV->getTruncInst()) ? nullptr : WideIV;
  }

  // Check if VPV is an optimizable induction increment.
  VPRecipeBase *Def = VPV->getDefiningRecipe();
  if (!Def || Def->getNumOperands() != 2)
    return nullptr;
  WideIV = dyn_cast<VPWidenInductionRecipe>(Def->getOperand(0));
  if (!WideIV)
    WideIV = dyn_cast<VPWidenInductionRecipe>(Def->getOperand(1));
  if (!WideIV)
    return nullptr;

  auto IsWideIVInc = [&]() {
    using namespace VPlanPatternMatch;
    auto &ID = WideIV->getInductionDescriptor();

    // Check if VPV increments the induction by the induction step.
    VPValue *IVStep = WideIV->getStepValue();
    switch (ID.getInductionOpcode()) {
    case Instruction::Add:
      return match(VPV, m_c_Binary<Instruction::Add>(m_Specific(WideIV),
                                                     m_Specific(IVStep)));
    case Instruction::FAdd:
      return match(VPV, m_c_Binary<Instruction::FAdd>(m_Specific(WideIV),
                                                      m_Specific(IVStep)));
    case Instruction::FSub:
      return match(VPV, m_Binary<Instruction::FSub>(m_Specific(WideIV),
                                                    m_Specific(IVStep)));
    case Instruction::Sub: {
      // IVStep will be the negated step of the subtraction. Check if Step == -1
      // * IVStep.
      VPValue *Step;
      if (!match(VPV,
                 m_Binary<Instruction::Sub>(m_VPValue(), m_VPValue(Step))) ||
          !Step->isLiveIn() || !IVStep->isLiveIn())
        return false;
      auto *StepCI = dyn_cast<ConstantInt>(Step->getLiveInIRValue());
      auto *IVStepCI = dyn_cast<ConstantInt>(IVStep->getLiveInIRValue());
      return StepCI && IVStepCI &&
             StepCI->getValue() == (-1 * IVStepCI->getValue());
    }
    default:
      return ID.getKind() == InductionDescriptor::IK_PtrInduction &&
             match(VPV, m_GetElementPtr(m_Specific(WideIV),
                                        m_Specific(WideIV->getStepValue())));
    }
    llvm_unreachable("should have been covered by switch above");
  };
  return IsWideIVInc() ? WideIV : nullptr;
}

void VPlanTransforms::optimizeInductionExitUsers(
    VPlan &Plan, DenseMap<VPValue *, VPValue *> &EndValues) {
  using namespace VPlanPatternMatch;
  SmallVector<VPIRBasicBlock *> ExitVPBBs(Plan.getExitBlocks());
  if (ExitVPBBs.size() != 1)
    return;

  VPIRBasicBlock *ExitVPBB = ExitVPBBs[0];
  VPBlockBase *PredVPBB = ExitVPBB->getSinglePredecessor();
  if (!PredVPBB)
    return;
  assert(PredVPBB == Plan.getMiddleBlock() &&
         "predecessor must be the middle block");

  VPTypeAnalysis TypeInfo(Plan.getCanonicalIV()->getScalarType());
  VPBuilder B(Plan.getMiddleBlock()->getTerminator());
  for (VPRecipeBase &R : *ExitVPBB) {
    auto *ExitIRI = cast<VPIRInstruction>(&R);
    if (!isa<PHINode>(ExitIRI->getInstruction()))
      break;

    VPValue *Incoming;
    if (!match(ExitIRI->getOperand(0),
               m_VPInstruction<VPInstruction::ExtractFromEnd>(
                   m_VPValue(Incoming), m_SpecificInt(1))))
      continue;

    auto *WideIV = getOptimizableIVOf(Incoming);
    if (!WideIV)
      continue;
    VPValue *EndValue = EndValues.lookup(WideIV);
    assert(EndValue && "end value must have been pre-computed");

    if (Incoming != WideIV) {
      ExitIRI->setOperand(0, EndValue);
      continue;
    }

    VPValue *Escape = nullptr;
    VPValue *Step = WideIV->getStepValue();
    Type *ScalarTy = TypeInfo.inferScalarType(WideIV);
    if (ScalarTy->isIntegerTy()) {
      Escape =
          B.createNaryOp(Instruction::Sub, {EndValue, Step}, {}, "ind.escape");
    } else if (ScalarTy->isPointerTy()) {
      auto *Zero = Plan.getOrAddLiveIn(
          ConstantInt::get(Step->getLiveInIRValue()->getType(), 0));
      Escape = B.createPtrAdd(EndValue,
                              B.createNaryOp(Instruction::Sub, {Zero, Step}),
                              {}, "ind.escape");
    } else if (ScalarTy->isFloatingPointTy()) {
      const auto &ID = WideIV->getInductionDescriptor();
      Escape = B.createNaryOp(
          ID.getInductionBinOp()->getOpcode() == Instruction::FAdd
              ? Instruction::FSub
              : Instruction::FAdd,
          {EndValue, Step}, {ID.getInductionBinOp()->getFastMathFlags()});
    } else {
      llvm_unreachable("all possible induction types must be handled");
    }
    ExitIRI->setOperand(0, Escape);
  }
}

/// Remove redundant EpxandSCEVRecipes in \p Plan's entry block by replacing
/// them with already existing recipes expanding the same SCEV expression.
static void removeRedundantExpandSCEVRecipes(VPlan &Plan) {
  DenseMap<const SCEV *, VPValue *> SCEV2VPV;

  for (VPRecipeBase &R :
       make_early_inc_range(*Plan.getEntry()->getEntryBasicBlock())) {
    auto *ExpR = dyn_cast<VPExpandSCEVRecipe>(&R);
    if (!ExpR)
      continue;

    auto I = SCEV2VPV.insert({ExpR->getSCEV(), ExpR});
    if (I.second)
      continue;
    ExpR->replaceAllUsesWith(I.first->second);
    ExpR->eraseFromParent();
  }
}

static void recursivelyDeleteDeadRecipes(VPValue *V) {
  SmallVector<VPValue *> WorkList;
  SmallPtrSet<VPValue *, 8> Seen;
  WorkList.push_back(V);

  while (!WorkList.empty()) {
    VPValue *Cur = WorkList.pop_back_val();
    if (!Seen.insert(Cur).second)
      continue;
    VPRecipeBase *R = Cur->getDefiningRecipe();
    if (!R)
      continue;
    if (!isDeadRecipe(*R))
      continue;
    WorkList.append(R->op_begin(), R->op_end());
    R->eraseFromParent();
  }
}

/// Try to simplify recipe \p R.
static void simplifyRecipe(VPRecipeBase &R, VPTypeAnalysis &TypeInfo) {
  using namespace llvm::VPlanPatternMatch;

  if (auto *Blend = dyn_cast<VPBlendRecipe>(&R)) {
    // Try to remove redundant blend recipes.
    SmallPtrSet<VPValue *, 4> UniqueValues;
    if (Blend->isNormalized() || !match(Blend->getMask(0), m_False()))
      UniqueValues.insert(Blend->getIncomingValue(0));
    for (unsigned I = 1; I != Blend->getNumIncomingValues(); ++I)
      if (!match(Blend->getMask(I), m_False()))
        UniqueValues.insert(Blend->getIncomingValue(I));

    if (UniqueValues.size() == 1) {
      Blend->replaceAllUsesWith(*UniqueValues.begin());
      Blend->eraseFromParent();
      return;
    }

    if (Blend->isNormalized())
      return;

    // Normalize the blend so its first incoming value is used as the initial
    // value with the others blended into it.

    unsigned StartIndex = 0;
    for (unsigned I = 0; I != Blend->getNumIncomingValues(); ++I) {
      // If a value's mask is used only by the blend then is can be deadcoded.
      // TODO: Find the most expensive mask that can be deadcoded, or a mask
      // that's used by multiple blends where it can be removed from them all.
      VPValue *Mask = Blend->getMask(I);
      if (Mask->getNumUsers() == 1 && !match(Mask, m_False())) {
        StartIndex = I;
        break;
      }
    }

    SmallVector<VPValue *, 4> OperandsWithMask;
    OperandsWithMask.push_back(Blend->getIncomingValue(StartIndex));

    for (unsigned I = 0; I != Blend->getNumIncomingValues(); ++I) {
      if (I == StartIndex)
        continue;
      OperandsWithMask.push_back(Blend->getIncomingValue(I));
      OperandsWithMask.push_back(Blend->getMask(I));
    }

    auto *NewBlend = new VPBlendRecipe(
        cast<PHINode>(Blend->getUnderlyingValue()), OperandsWithMask);
    NewBlend->insertBefore(&R);

    VPValue *DeadMask = Blend->getMask(StartIndex);
    Blend->replaceAllUsesWith(NewBlend);
    Blend->eraseFromParent();
    recursivelyDeleteDeadRecipes(DeadMask);

    /// Simplify BLEND %a, %b, Not(%mask) -> BLEND %b, %a, %mask.
    VPValue *NewMask;
    if (NewBlend->getNumOperands() == 3 &&
        match(NewBlend->getMask(1), m_Not(m_VPValue(NewMask)))) {
      VPValue *Inc0 = NewBlend->getOperand(0);
      VPValue *Inc1 = NewBlend->getOperand(1);
      VPValue *OldMask = NewBlend->getOperand(2);
      NewBlend->setOperand(0, Inc1);
      NewBlend->setOperand(1, Inc0);
      NewBlend->setOperand(2, NewMask);
      if (OldMask->getNumUsers() == 0)
        cast<VPInstruction>(OldMask)->eraseFromParent();
    }
    return;
  }

  VPValue *A;
  if (match(&R, m_Trunc(m_ZExtOrSExt(m_VPValue(A))))) {
    VPValue *Trunc = R.getVPSingleValue();
    Type *TruncTy = TypeInfo.inferScalarType(Trunc);
    Type *ATy = TypeInfo.inferScalarType(A);
    if (TruncTy == ATy) {
      Trunc->replaceAllUsesWith(A);
    } else {
      // Don't replace a scalarizing recipe with a widened cast.
      if (isa<VPReplicateRecipe>(&R))
        return;
      if (ATy->getScalarSizeInBits() < TruncTy->getScalarSizeInBits()) {

        unsigned ExtOpcode = match(R.getOperand(0), m_SExt(m_VPValue()))
                                 ? Instruction::SExt
                                 : Instruction::ZExt;
        auto *VPC =
            new VPWidenCastRecipe(Instruction::CastOps(ExtOpcode), A, TruncTy);
        if (auto *UnderlyingExt = R.getOperand(0)->getUnderlyingValue()) {
          // UnderlyingExt has distinct return type, used to retain legacy cost.
          VPC->setUnderlyingValue(UnderlyingExt);
        }
        VPC->insertBefore(&R);
        Trunc->replaceAllUsesWith(VPC);
      } else if (ATy->getScalarSizeInBits() > TruncTy->getScalarSizeInBits()) {
        auto *VPC = new VPWidenCastRecipe(Instruction::Trunc, A, TruncTy);
        VPC->insertBefore(&R);
        Trunc->replaceAllUsesWith(VPC);
      }
    }
#ifndef NDEBUG
    // Verify that the cached type info is for both A and its users is still
    // accurate by comparing it to freshly computed types.
    VPTypeAnalysis TypeInfo2(
        R.getParent()->getPlan()->getCanonicalIV()->getScalarType());
    assert(TypeInfo.inferScalarType(A) == TypeInfo2.inferScalarType(A));
    for (VPUser *U : A->users()) {
      auto *R = cast<VPRecipeBase>(U);
      for (VPValue *VPV : R->definedValues())
        assert(TypeInfo.inferScalarType(VPV) == TypeInfo2.inferScalarType(VPV));
    }
#endif
  }

  // Simplify (X && Y) || (X && !Y) -> X.
  // TODO: Split up into simpler, modular combines: (X && Y) || (X && Z) into X
  // && (Y || Z) and (X || !X) into true. This requires queuing newly created
  // recipes to be visited during simplification.
  VPValue *X, *Y, *X1, *Y1;
  if (match(&R,
            m_c_BinaryOr(m_LogicalAnd(m_VPValue(X), m_VPValue(Y)),
                         m_LogicalAnd(m_VPValue(X1), m_Not(m_VPValue(Y1))))) &&
      X == X1 && Y == Y1) {
    R.getVPSingleValue()->replaceAllUsesWith(X);
    R.eraseFromParent();
    return;
  }

  if (match(&R, m_c_Mul(m_VPValue(A), m_SpecificInt(1))))
    return R.getVPSingleValue()->replaceAllUsesWith(A);

  if (match(&R, m_Not(m_Not(m_VPValue(A)))))
    return R.getVPSingleValue()->replaceAllUsesWith(A);

  // Remove redundant DerviedIVs, that is 0 + A * 1 -> A and 0 + 0 * x -> 0.
  if ((match(&R,
             m_DerivedIV(m_SpecificInt(0), m_VPValue(A), m_SpecificInt(1))) ||
       match(&R,
             m_DerivedIV(m_SpecificInt(0), m_SpecificInt(0), m_VPValue()))) &&
      TypeInfo.inferScalarType(R.getOperand(1)) ==
          TypeInfo.inferScalarType(R.getVPSingleValue()))
    return R.getVPSingleValue()->replaceAllUsesWith(R.getOperand(1));
}

void VPlanTransforms::simplifyRecipes(VPlan &Plan, Type &CanonicalIVTy) {
  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>> RPOT(
      Plan.getEntry());
  VPTypeAnalysis TypeInfo(&CanonicalIVTy);
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT)) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      simplifyRecipe(R, TypeInfo);
    }
  }
}

void VPlanTransforms::optimizeForVFAndUF(VPlan &Plan, ElementCount BestVF,
                                         unsigned BestUF,
                                         PredicatedScalarEvolution &PSE) {
  assert(Plan.hasVF(BestVF) && "BestVF is not available in Plan");
  assert(Plan.hasUF(BestUF) && "BestUF is not available in Plan");
  VPRegionBlock *VectorRegion = Plan.getVectorLoopRegion();
  VPBasicBlock *ExitingVPBB = VectorRegion->getExitingBasicBlock();
  auto *Term = &ExitingVPBB->back();
  // Try to simplify the branch condition if TC <= VF * UF when preparing to
  // execute the plan for the main vector loop. We only do this if the
  // terminator is:
  //  1. BranchOnCount, or
  //  2. BranchOnCond where the input is Not(ActiveLaneMask).
  using namespace llvm::VPlanPatternMatch;
  if (!match(Term, m_BranchOnCount(m_VPValue(), m_VPValue())) &&
      !match(Term,
             m_BranchOnCond(m_Not(m_ActiveLaneMask(m_VPValue(), m_VPValue())))))
    return;

  ScalarEvolution &SE = *PSE.getSE();
  const SCEV *TripCount =
      vputils::getSCEVExprForVPValue(Plan.getTripCount(), SE);
  assert(!isa<SCEVCouldNotCompute>(TripCount) &&
         "Trip count SCEV must be computable");
  ElementCount NumElements = BestVF.multiplyCoefficientBy(BestUF);
  const SCEV *C = SE.getElementCount(TripCount->getType(), NumElements);
  if (TripCount->isZero() ||
      !SE.isKnownPredicate(CmpInst::ICMP_ULE, TripCount, C))
    return;

  // The vector loop region only executes once. If possible, completely remove
  // the region, otherwise replace the terminator controlling the latch with
  // (BranchOnCond true).
  auto *Header = cast<VPBasicBlock>(VectorRegion->getEntry());
  auto *CanIVTy = Plan.getCanonicalIV()->getScalarType();
  if (all_of(
          Header->phis(),
          IsaPred<VPCanonicalIVPHIRecipe, VPFirstOrderRecurrencePHIRecipe>)) {
    for (VPRecipeBase &HeaderR : make_early_inc_range(Header->phis())) {
      auto *HeaderPhiR = cast<VPHeaderPHIRecipe>(&HeaderR);
      HeaderPhiR->replaceAllUsesWith(HeaderPhiR->getStartValue());
      HeaderPhiR->eraseFromParent();
    }

    VPBlockBase *Preheader = VectorRegion->getSinglePredecessor();
    VPBlockBase *Exit = VectorRegion->getSingleSuccessor();
    VPBlockUtils::disconnectBlocks(Preheader, VectorRegion);
    VPBlockUtils::disconnectBlocks(VectorRegion, Exit);

    for (VPBlockBase *B : vp_depth_first_shallow(VectorRegion->getEntry()))
      B->setParent(nullptr);

    VPBlockUtils::connectBlocks(Preheader, Header);
    VPBlockUtils::connectBlocks(ExitingVPBB, Exit);
    simplifyRecipes(Plan, *CanIVTy);
  } else {
    // The vector region contains header phis for which we cannot remove the
    // loop region yet.
    LLVMContext &Ctx = SE.getContext();
    auto *BOC = new VPInstruction(
        VPInstruction::BranchOnCond,
        {Plan.getOrAddLiveIn(ConstantInt::getTrue(Ctx))}, Term->getDebugLoc());
    ExitingVPBB->appendRecipe(BOC);
  }

  Term->eraseFromParent();

  Plan.setVF(BestVF);
  Plan.setUF(BestUF);
  // TODO: Further simplifications are possible
  //      1. Replace inductions with constants.
  //      2. Replace vector loop region with VPBasicBlock.
}

/// Sink users of \p FOR after the recipe defining the previous value \p
/// Previous of the recurrence. \returns true if all users of \p FOR could be
/// re-arranged as needed or false if it is not possible.
static bool
sinkRecurrenceUsersAfterPrevious(VPFirstOrderRecurrencePHIRecipe *FOR,
                                 VPRecipeBase *Previous,
                                 VPDominatorTree &VPDT) {
  // Collect recipes that need sinking.
  SmallVector<VPRecipeBase *> WorkList;
  SmallPtrSet<VPRecipeBase *, 8> Seen;
  Seen.insert(Previous);
  auto TryToPushSinkCandidate = [&](VPRecipeBase *SinkCandidate) {
    // The previous value must not depend on the users of the recurrence phi. In
    // that case, FOR is not a fixed order recurrence.
    if (SinkCandidate == Previous)
      return false;

    if (isa<VPHeaderPHIRecipe>(SinkCandidate) ||
        !Seen.insert(SinkCandidate).second ||
        VPDT.properlyDominates(Previous, SinkCandidate))
      return true;

    if (SinkCandidate->mayHaveSideEffects())
      return false;

    WorkList.push_back(SinkCandidate);
    return true;
  };

  // Recursively sink users of FOR after Previous.
  WorkList.push_back(FOR);
  for (unsigned I = 0; I != WorkList.size(); ++I) {
    VPRecipeBase *Current = WorkList[I];
    assert(Current->getNumDefinedValues() == 1 &&
           "only recipes with a single defined value expected");

    for (VPUser *User : Current->getVPSingleValue()->users()) {
      if (!TryToPushSinkCandidate(cast<VPRecipeBase>(User)))
        return false;
    }
  }

  // Keep recipes to sink ordered by dominance so earlier instructions are
  // processed first.
  sort(WorkList, [&VPDT](const VPRecipeBase *A, const VPRecipeBase *B) {
    return VPDT.properlyDominates(A, B);
  });

  for (VPRecipeBase *SinkCandidate : WorkList) {
    if (SinkCandidate == FOR)
      continue;

    SinkCandidate->moveAfter(Previous);
    Previous = SinkCandidate;
  }
  return true;
}

/// Try to hoist \p Previous and its operands before all users of \p FOR.
static bool hoistPreviousBeforeFORUsers(VPFirstOrderRecurrencePHIRecipe *FOR,
                                        VPRecipeBase *Previous,
                                        VPDominatorTree &VPDT) {
  if (Previous->mayHaveSideEffects() || Previous->mayReadFromMemory())
    return false;

  // Collect recipes that need hoisting.
  SmallVector<VPRecipeBase *> HoistCandidates;
  SmallPtrSet<VPRecipeBase *, 8> Visited;
  VPRecipeBase *HoistPoint = nullptr;
  // Find the closest hoist point by looking at all users of FOR and selecting
  // the recipe dominating all other users.
  for (VPUser *U : FOR->users()) {
    auto *R = cast<VPRecipeBase>(U);
    if (!HoistPoint || VPDT.properlyDominates(R, HoistPoint))
      HoistPoint = R;
  }
  assert(all_of(FOR->users(),
                [&VPDT, HoistPoint](VPUser *U) {
                  auto *R = cast<VPRecipeBase>(U);
                  return HoistPoint == R ||
                         VPDT.properlyDominates(HoistPoint, R);
                }) &&
         "HoistPoint must dominate all users of FOR");

  auto NeedsHoisting = [HoistPoint, &VPDT,
                        &Visited](VPValue *HoistCandidateV) -> VPRecipeBase * {
    VPRecipeBase *HoistCandidate = HoistCandidateV->getDefiningRecipe();
    if (!HoistCandidate)
      return nullptr;
    VPRegionBlock *EnclosingLoopRegion =
        HoistCandidate->getParent()->getEnclosingLoopRegion();
    assert((!HoistCandidate->getParent()->getParent() ||
            HoistCandidate->getParent()->getParent() == EnclosingLoopRegion) &&
           "CFG in VPlan should still be flat, without replicate regions");
    // Hoist candidate was already visited, no need to hoist.
    if (!Visited.insert(HoistCandidate).second)
      return nullptr;

    // Candidate is outside loop region or a header phi, dominates FOR users w/o
    // hoisting.
    if (!EnclosingLoopRegion || isa<VPHeaderPHIRecipe>(HoistCandidate))
      return nullptr;

    // If we reached a recipe that dominates HoistPoint, we don't need to
    // hoist the recipe.
    if (VPDT.properlyDominates(HoistCandidate, HoistPoint))
      return nullptr;
    return HoistCandidate;
  };
  auto CanHoist = [&](VPRecipeBase *HoistCandidate) {
    // Avoid hoisting candidates with side-effects, as we do not yet analyze
    // associated dependencies.
    return !HoistCandidate->mayHaveSideEffects();
  };

  if (!NeedsHoisting(Previous->getVPSingleValue()))
    return true;

  // Recursively try to hoist Previous and its operands before all users of FOR.
  HoistCandidates.push_back(Previous);

  for (unsigned I = 0; I != HoistCandidates.size(); ++I) {
    VPRecipeBase *Current = HoistCandidates[I];
    assert(Current->getNumDefinedValues() == 1 &&
           "only recipes with a single defined value expected");
    if (!CanHoist(Current))
      return false;

    for (VPValue *Op : Current->operands()) {
      // If we reach FOR, it means the original Previous depends on some other
      // recurrence that in turn depends on FOR. If that is the case, we would
      // also need to hoist recipes involving the other FOR, which may break
      // dependencies.
      if (Op == FOR)
        return false;

      if (auto *R = NeedsHoisting(Op))
        HoistCandidates.push_back(R);
    }
  }

  // Order recipes to hoist by dominance so earlier instructions are processed
  // first.
  sort(HoistCandidates, [&VPDT](const VPRecipeBase *A, const VPRecipeBase *B) {
    return VPDT.properlyDominates(A, B);
  });

  for (VPRecipeBase *HoistCandidate : HoistCandidates) {
    HoistCandidate->moveBefore(*HoistPoint->getParent(),
                               HoistPoint->getIterator());
  }

  return true;
}

bool VPlanTransforms::adjustFixedOrderRecurrences(VPlan &Plan,
                                                  VPBuilder &LoopBuilder) {
  VPDominatorTree VPDT;
  VPDT.recalculate(Plan);

  SmallVector<VPFirstOrderRecurrencePHIRecipe *> RecurrencePhis;
  for (VPRecipeBase &R :
       Plan.getVectorLoopRegion()->getEntry()->getEntryBasicBlock()->phis())
    if (auto *FOR = dyn_cast<VPFirstOrderRecurrencePHIRecipe>(&R))
      RecurrencePhis.push_back(FOR);

  for (VPFirstOrderRecurrencePHIRecipe *FOR : RecurrencePhis) {
    SmallPtrSet<VPFirstOrderRecurrencePHIRecipe *, 4> SeenPhis;
    VPRecipeBase *Previous = FOR->getBackedgeValue()->getDefiningRecipe();
    // Fixed-order recurrences do not contain cycles, so this loop is guaranteed
    // to terminate.
    while (auto *PrevPhi =
               dyn_cast_or_null<VPFirstOrderRecurrencePHIRecipe>(Previous)) {
      assert(PrevPhi->getParent() == FOR->getParent());
      assert(SeenPhis.insert(PrevPhi).second);
      Previous = PrevPhi->getBackedgeValue()->getDefiningRecipe();
    }

    if (!sinkRecurrenceUsersAfterPrevious(FOR, Previous, VPDT) &&
        !hoistPreviousBeforeFORUsers(FOR, Previous, VPDT))
      return false;

    // Introduce a recipe to combine the incoming and previous values of a
    // fixed-order recurrence.
    VPBasicBlock *InsertBlock = Previous->getParent();
    if (isa<VPHeaderPHIRecipe>(Previous))
      LoopBuilder.setInsertPoint(InsertBlock, InsertBlock->getFirstNonPhi());
    else
      LoopBuilder.setInsertPoint(InsertBlock,
                                 std::next(Previous->getIterator()));

    auto *RecurSplice =
        LoopBuilder.createNaryOp(VPInstruction::FirstOrderRecurrenceSplice,
                                 {FOR, FOR->getBackedgeValue()});

    FOR->replaceAllUsesWith(RecurSplice);
    // Set the first operand of RecurSplice to FOR again, after replacing
    // all users.
    RecurSplice->setOperand(0, FOR);
  }
  return true;
}

void VPlanTransforms::clearReductionWrapFlags(VPlan &Plan) {
  for (VPRecipeBase &R :
       Plan.getVectorLoopRegion()->getEntryBasicBlock()->phis()) {
    auto *PhiR = dyn_cast<VPReductionPHIRecipe>(&R);
    if (!PhiR)
      continue;
    const RecurrenceDescriptor &RdxDesc = PhiR->getRecurrenceDescriptor();
    RecurKind RK = RdxDesc.getRecurrenceKind();
    if (RK != RecurKind::Add && RK != RecurKind::Mul)
      continue;

    for (VPUser *U : collectUsersRecursively(PhiR))
      if (auto *RecWithFlags = dyn_cast<VPRecipeWithIRFlags>(U)) {
        RecWithFlags->dropPoisonGeneratingFlags();
      }
  }
}

/// Move loop-invariant recipes out of the vector loop region in \p Plan.
static void licm(VPlan &Plan) {
  VPBasicBlock *Preheader = Plan.getVectorPreheader();

  // Return true if we do not know how to (mechanically) hoist a given recipe
  // out of a loop region. Does not address legality concerns such as aliasing
  // or speculation safety.
  auto CannotHoistRecipe = [](VPRecipeBase &R) {
    // Allocas cannot be hoisted.
    auto *RepR = dyn_cast<VPReplicateRecipe>(&R);
    return RepR && RepR->getOpcode() == Instruction::Alloca;
  };

  // Hoist any loop invariant recipes from the vector loop region to the
  // preheader. Preform a shallow traversal of the vector loop region, to
  // exclude recipes in replicate regions.
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_shallow(LoopRegion->getEntry()))) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      if (CannotHoistRecipe(R))
        continue;
      // TODO: Relax checks in the future, e.g. we could also hoist reads, if
      // their memory location is not modified in the vector loop.
      if (R.mayHaveSideEffects() || R.mayReadFromMemory() || R.isPhi() ||
          any_of(R.operands(), [](VPValue *Op) {
            return !Op->isDefinedOutsideLoopRegions();
          }))
        continue;
      R.moveBefore(*Preheader, Preheader->end());
    }
  }
}

void VPlanTransforms::truncateToMinimalBitwidths(
    VPlan &Plan, const MapVector<Instruction *, uint64_t> &MinBWs) {
#ifndef NDEBUG
  // Count the processed recipes and cross check the count later with MinBWs
  // size, to make sure all entries in MinBWs have been handled.
  unsigned NumProcessedRecipes = 0;
#endif
  // Keep track of created truncates, so they can be re-used. Note that we
  // cannot use RAUW after creating a new truncate, as this would could make
  // other uses have different types for their operands, making them invalidly
  // typed.
  DenseMap<VPValue *, VPWidenCastRecipe *> ProcessedTruncs;
  Type *CanonicalIVType = Plan.getCanonicalIV()->getScalarType();
  VPTypeAnalysis TypeInfo(CanonicalIVType);
  VPBasicBlock *PH = Plan.getVectorPreheader();
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getVectorLoopRegion()))) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      if (!isa<VPWidenRecipe, VPWidenCastRecipe, VPReplicateRecipe,
               VPWidenSelectRecipe, VPWidenLoadRecipe>(&R))
        continue;

      VPValue *ResultVPV = R.getVPSingleValue();
      auto *UI = cast_or_null<Instruction>(ResultVPV->getUnderlyingValue());
      unsigned NewResSizeInBits = MinBWs.lookup(UI);
      if (!NewResSizeInBits)
        continue;

#ifndef NDEBUG
      NumProcessedRecipes++;
#endif
      // If the value wasn't vectorized, we must maintain the original scalar
      // type. Skip those here, after incrementing NumProcessedRecipes. Also
      // skip casts which do not need to be handled explicitly here, as
      // redundant casts will be removed during recipe simplification.
      if (isa<VPReplicateRecipe, VPWidenCastRecipe>(&R)) {
#ifndef NDEBUG
        // If any of the operands is a live-in and not used by VPWidenRecipe or
        // VPWidenSelectRecipe, but in MinBWs, make sure it is counted as
        // processed as well. When MinBWs is currently constructed, there is no
        // information about whether recipes are widened or replicated and in
        // case they are reciplicated the operands are not truncated. Counting
        // them them here ensures we do not miss any recipes in MinBWs.
        // TODO: Remove once the analysis is done on VPlan.
        for (VPValue *Op : R.operands()) {
          if (!Op->isLiveIn())
            continue;
          auto *UV = dyn_cast_or_null<Instruction>(Op->getUnderlyingValue());
          if (UV && MinBWs.contains(UV) && !ProcessedTruncs.contains(Op) &&
              none_of(Op->users(),
                      IsaPred<VPWidenRecipe, VPWidenSelectRecipe>)) {
            // Add an entry to ProcessedTruncs to avoid counting the same
            // operand multiple times.
            ProcessedTruncs[Op] = nullptr;
            NumProcessedRecipes += 1;
          }
        }
#endif
        continue;
      }

      Type *OldResTy = TypeInfo.inferScalarType(ResultVPV);
      unsigned OldResSizeInBits = OldResTy->getScalarSizeInBits();
      assert(OldResTy->isIntegerTy() && "only integer types supported");
      (void)OldResSizeInBits;

      LLVMContext &Ctx = CanonicalIVType->getContext();
      auto *NewResTy = IntegerType::get(Ctx, NewResSizeInBits);

      // Any wrapping introduced by shrinking this operation shouldn't be
      // considered undefined behavior. So, we can't unconditionally copy
      // arithmetic wrapping flags to VPW.
      if (auto *VPW = dyn_cast<VPRecipeWithIRFlags>(&R))
        VPW->dropPoisonGeneratingFlags();

      using namespace llvm::VPlanPatternMatch;
      if (OldResSizeInBits != NewResSizeInBits &&
          !match(&R, m_Binary<Instruction::ICmp>(m_VPValue(), m_VPValue()))) {
        // Extend result to original width.
        auto *Ext =
            new VPWidenCastRecipe(Instruction::ZExt, ResultVPV, OldResTy);
        Ext->insertAfter(&R);
        ResultVPV->replaceAllUsesWith(Ext);
        Ext->setOperand(0, ResultVPV);
        assert(OldResSizeInBits > NewResSizeInBits && "Nothing to shrink?");
      } else {
        assert(
            match(&R, m_Binary<Instruction::ICmp>(m_VPValue(), m_VPValue())) &&
            "Only ICmps should not need extending the result.");
      }

      assert(!isa<VPWidenStoreRecipe>(&R) && "stores cannot be narrowed");
      if (isa<VPWidenLoadRecipe>(&R))
        continue;

      // Shrink operands by introducing truncates as needed.
      unsigned StartIdx = isa<VPWidenSelectRecipe>(&R) ? 1 : 0;
      for (unsigned Idx = StartIdx; Idx != R.getNumOperands(); ++Idx) {
        auto *Op = R.getOperand(Idx);
        unsigned OpSizeInBits =
            TypeInfo.inferScalarType(Op)->getScalarSizeInBits();
        if (OpSizeInBits == NewResSizeInBits)
          continue;
        assert(OpSizeInBits > NewResSizeInBits && "nothing to truncate");
        auto [ProcessedIter, IterIsEmpty] =
            ProcessedTruncs.insert({Op, nullptr});
        VPWidenCastRecipe *NewOp =
            IterIsEmpty
                ? new VPWidenCastRecipe(Instruction::Trunc, Op, NewResTy)
                : ProcessedIter->second;
        R.setOperand(Idx, NewOp);
        if (!IterIsEmpty)
          continue;
        ProcessedIter->second = NewOp;
        if (!Op->isLiveIn()) {
          NewOp->insertBefore(&R);
        } else {
          PH->appendRecipe(NewOp);
#ifndef NDEBUG
          auto *OpInst = dyn_cast<Instruction>(Op->getLiveInIRValue());
          bool IsContained = MinBWs.contains(OpInst);
          NumProcessedRecipes += IsContained;
#endif
        }
      }

    }
  }

  assert(MinBWs.size() == NumProcessedRecipes &&
         "some entries in MinBWs haven't been processed");
}

void VPlanTransforms::optimize(VPlan &Plan) {
  runPass(removeRedundantCanonicalIVs, Plan);
  runPass(removeRedundantInductionCasts, Plan);

  runPass(simplifyRecipes, Plan, *Plan.getCanonicalIV()->getScalarType());
  runPass(removeDeadRecipes, Plan);
  runPass(legalizeAndOptimizeInductions, Plan);
  runPass(optimizeInnerLoopInductions, Plan);
  runPass(removeRedundantExpandSCEVRecipes, Plan);
  runPass(simplifyRecipes, Plan, *Plan.getCanonicalIV()->getScalarType());
  runPass(removeDeadRecipes, Plan);

  if (!Plan.hasVF(ElementCount::getFixed(1)))
    runPass(handleMaskedUniformReplicateRecipes, Plan);
  runPass(createAndOptimizeReplicateRegions, Plan);
  runPass(mergeBlocksIntoPredecessors, Plan);
  runPass(licm, Plan);
}

// Add a VPActiveLaneMaskPHIRecipe and related recipes to \p Plan and replace
// the loop terminator with a branch-on-cond recipe with the negated
// active-lane-mask as operand. Note that this turns the loop into an
// uncountable one. Only the existing terminator is replaced, all other existing
// recipes/users remain unchanged, except for poison-generating flags being
// dropped from the canonical IV increment. Return the created
// VPActiveLaneMaskPHIRecipe.
//
// The function uses the following definitions:
//
//  %TripCount = DataWithControlFlowWithoutRuntimeCheck ?
//    calculate-trip-count-minus-VF (original TC) : original TC
//  %IncrementValue = DataWithControlFlowWithoutRuntimeCheck ?
//     CanonicalIVPhi : CanonicalIVIncrement
//  %StartV is the canonical induction start value.
//
// The function adds the following recipes:
//
// vector.ph:
//   %TripCount = calculate-trip-count-minus-VF (original TC)
//       [if DataWithControlFlowWithoutRuntimeCheck]
//   %EntryInc = canonical-iv-increment-for-part %StartV
//   %EntryALM = active-lane-mask %EntryInc, %TripCount
//
// vector.body:
//   ...
//   %P = active-lane-mask-phi [ %EntryALM, %vector.ph ], [ %ALM, %vector.body ]
//   ...
//   %InLoopInc = canonical-iv-increment-for-part %IncrementValue
//   %ALM = active-lane-mask %InLoopInc, TripCount
//   %Negated = Not %ALM
//   branch-on-cond %Negated
//
static VPActiveLaneMaskPHIRecipe *addVPLaneMaskPhiAndUpdateExitBranch(
    VPlan &Plan, bool DataAndControlFlowWithoutRuntimeCheck) {
  VPRegionBlock *TopRegion = Plan.getVectorLoopRegion();
  VPBasicBlock *EB = TopRegion->getExitingBasicBlock();
  auto *CanonicalIVPHI = Plan.getCanonicalIV();
  VPValue *StartV = CanonicalIVPHI->getStartValue();

  auto *CanonicalIVIncrement =
      cast<VPInstruction>(CanonicalIVPHI->getBackedgeValue());
  // TODO: Check if dropping the flags is needed if
  // !DataAndControlFlowWithoutRuntimeCheck.
  CanonicalIVIncrement->dropPoisonGeneratingFlags();
  DebugLoc DL = CanonicalIVIncrement->getDebugLoc();
  // We can't use StartV directly in the ActiveLaneMask VPInstruction, since
  // we have to take unrolling into account. Each part needs to start at
  //   Part * VF
  auto *VecPreheader = Plan.getVectorPreheader();
  VPBuilder Builder(VecPreheader);

  // Create the ActiveLaneMask instruction using the correct start values.
  VPValue *TC = Plan.getTripCount();

  VPValue *TripCount, *IncrementValue;
  if (!DataAndControlFlowWithoutRuntimeCheck) {
    // When the loop is guarded by a runtime overflow check for the loop
    // induction variable increment by VF, we can increment the value before
    // the get.active.lane mask and use the unmodified tripcount.
    IncrementValue = CanonicalIVIncrement;
    TripCount = TC;
  } else {
    // When avoiding a runtime check, the active.lane.mask inside the loop
    // uses a modified trip count and the induction variable increment is
    // done after the active.lane.mask intrinsic is called.
    IncrementValue = CanonicalIVPHI;
    TripCount = Builder.createNaryOp(VPInstruction::CalculateTripCountMinusVF,
                                     {TC}, DL);
  }
  auto *EntryIncrement = Builder.createOverflowingOp(
      VPInstruction::CanonicalIVIncrementForPart, {StartV}, {false, false}, DL,
      "index.part.next");

  // Create the active lane mask instruction in the VPlan preheader.
  auto *EntryALM =
      Builder.createNaryOp(VPInstruction::ActiveLaneMask, {EntryIncrement, TC},
                           DL, "active.lane.mask.entry");

  // Now create the ActiveLaneMaskPhi recipe in the main loop using the
  // preheader ActiveLaneMask instruction.
  auto *LaneMaskPhi = new VPActiveLaneMaskPHIRecipe(EntryALM, DebugLoc());
  LaneMaskPhi->insertAfter(CanonicalIVPHI);

  // Create the active lane mask for the next iteration of the loop before the
  // original terminator.
  VPRecipeBase *OriginalTerminator = EB->getTerminator();
  Builder.setInsertPoint(OriginalTerminator);
  auto *InLoopIncrement =
      Builder.createOverflowingOp(VPInstruction::CanonicalIVIncrementForPart,
                                  {IncrementValue}, {false, false}, DL);
  auto *ALM = Builder.createNaryOp(VPInstruction::ActiveLaneMask,
                                   {InLoopIncrement, TripCount}, DL,
                                   "active.lane.mask.next");
  LaneMaskPhi->addOperand(ALM);

  // Replace the original terminator with BranchOnCond. We have to invert the
  // mask here because a true condition means jumping to the exit block.
  auto *NotMask = Builder.createNot(ALM, DL);
  Builder.createNaryOp(VPInstruction::BranchOnCond, {NotMask}, DL);
  OriginalTerminator->eraseFromParent();
  return LaneMaskPhi;
}

/// Collect all VPValues representing a header mask through the (ICMP_ULE,
/// WideCanonicalIV, backedge-taken-count) pattern.
/// TODO: Introduce explicit recipe for header-mask instead of searching
/// for the header-mask pattern manually.
static SmallVector<VPValue *> collectAllHeaderMasks(VPlan &Plan) {
  SmallVector<VPValue *> WideCanonicalIVs;
  auto *FoundWidenCanonicalIVUser =
      find_if(Plan.getCanonicalIV()->users(),
              [](VPUser *U) { return isa<VPWidenCanonicalIVRecipe>(U); });
  assert(count_if(Plan.getCanonicalIV()->users(),
                  [](VPUser *U) { return isa<VPWidenCanonicalIVRecipe>(U); }) <=
             1 &&
         "Must have at most one VPWideCanonicalIVRecipe");
  if (FoundWidenCanonicalIVUser != Plan.getCanonicalIV()->users().end()) {
    auto *WideCanonicalIV =
        cast<VPWidenCanonicalIVRecipe>(*FoundWidenCanonicalIVUser);
    WideCanonicalIVs.push_back(WideCanonicalIV);
  }

  // Also include VPWidenIntOrFpInductionRecipes that represent a widened
  // version of the canonical induction.
  VPBasicBlock *HeaderVPBB = Plan.getVectorLoopRegion()->getEntryBasicBlock();
  for (VPRecipeBase &Phi : HeaderVPBB->phis()) {
    auto *WidenOriginalIV = dyn_cast<VPWidenIntOrFpInductionRecipe>(&Phi);
    if (WidenOriginalIV && WidenOriginalIV->isCanonical())
      WideCanonicalIVs.push_back(WidenOriginalIV);
  }

  // Walk users of wide canonical IVs and collect to all compares of the form
  // (ICMP_ULE, WideCanonicalIV, backedge-taken-count).
  SmallVector<VPValue *> HeaderMasks;
  for (auto *Wide : WideCanonicalIVs) {
    for (VPUser *U : SmallVector<VPUser *>(Wide->users())) {
      auto *HeaderMask = dyn_cast<VPInstruction>(U);
      if (!HeaderMask || !vputils::isHeaderMask(HeaderMask, Plan))
        continue;

      assert(HeaderMask->getOperand(0) == Wide &&
             "WidenCanonicalIV must be the first operand of the compare");
      HeaderMasks.push_back(HeaderMask);
    }
  }
  return HeaderMasks;
}

void VPlanTransforms::addActiveLaneMask(
    VPlan &Plan, bool UseActiveLaneMaskForControlFlow,
    bool DataAndControlFlowWithoutRuntimeCheck) {
  assert((!DataAndControlFlowWithoutRuntimeCheck ||
          UseActiveLaneMaskForControlFlow) &&
         "DataAndControlFlowWithoutRuntimeCheck implies "
         "UseActiveLaneMaskForControlFlow");

  auto *FoundWidenCanonicalIVUser =
      find_if(Plan.getCanonicalIV()->users(),
              [](VPUser *U) { return isa<VPWidenCanonicalIVRecipe>(U); });
  assert(FoundWidenCanonicalIVUser &&
         "Must have widened canonical IV when tail folding!");
  auto *WideCanonicalIV =
      cast<VPWidenCanonicalIVRecipe>(*FoundWidenCanonicalIVUser);
  VPSingleDefRecipe *LaneMask;
  if (UseActiveLaneMaskForControlFlow) {
    LaneMask = addVPLaneMaskPhiAndUpdateExitBranch(
        Plan, DataAndControlFlowWithoutRuntimeCheck);
  } else {
    VPBuilder B = VPBuilder::getToInsertAfter(WideCanonicalIV);
    LaneMask = B.createNaryOp(VPInstruction::ActiveLaneMask,
                              {WideCanonicalIV, Plan.getTripCount()}, nullptr,
                              "active.lane.mask");
  }

  // Walk users of WideCanonicalIV and replace all compares of the form
  // (ICMP_ULE, WideCanonicalIV, backedge-taken-count) with an
  // active-lane-mask.
  for (VPValue *HeaderMask : collectAllHeaderMasks(Plan))
    HeaderMask->replaceAllUsesWith(LaneMask);
}

/// Try to convert \p CurRecipe to a corresponding EVL-based recipe. Returns
/// nullptr if no EVL-based recipe could be created.
/// \p HeaderMask  Header Mask.
/// \p CurRecipe   Recipe to be transform.
/// \p TypeInfo    VPlan-based type analysis.
/// \p AllOneMask  The vector mask parameter of vector-predication intrinsics.
/// \p EVL         The explicit vector length parameter of vector-predication
/// intrinsics.
/// \p PrevEVL     The explicit vector length of the previous iteration. Only
/// required if \p CurRecipe is a VPInstruction::FirstOrderRecurrenceSplice.
static VPRecipeBase *createEVLRecipe(VPValue *HeaderMask,
                                     VPRecipeBase &CurRecipe,
                                     VPTypeAnalysis &TypeInfo,
                                     VPValue &AllOneMask, VPValue &EVL,
                                     VPValue *PrevEVL) {
  using namespace llvm::VPlanPatternMatch;
  auto GetNewMask = [&](VPValue *OrigMask) -> VPValue * {
    assert(OrigMask && "Unmasked recipe when folding tail");
    return HeaderMask == OrigMask ? nullptr : OrigMask;
  };

  return TypeSwitch<VPRecipeBase *, VPRecipeBase *>(&CurRecipe)
      .Case<VPWidenLoadRecipe>([&](VPWidenLoadRecipe *L) {
        VPValue *NewMask = GetNewMask(L->getMask());
        return new VPWidenLoadEVLRecipe(*L, EVL, NewMask);
      })
      .Case<VPWidenStoreRecipe>([&](VPWidenStoreRecipe *S) {
        VPValue *NewMask = GetNewMask(S->getMask());
        return new VPWidenStoreEVLRecipe(*S, EVL, NewMask);
      })
      .Case<VPReductionRecipe>([&](VPReductionRecipe *Red) {
        VPValue *NewMask = GetNewMask(Red->getCondOp());
        return new VPReductionEVLRecipe(*Red, EVL, NewMask);
      })
      .Case<VPWidenSelectRecipe>([&](VPWidenSelectRecipe *Sel) {
        SmallVector<VPValue *> Ops(Sel->operands());
        Ops.push_back(&EVL);
        return new VPWidenIntrinsicRecipe(Intrinsic::vp_select, Ops,
                                          TypeInfo.inferScalarType(Sel),
                                          Sel->getDebugLoc());
      })
      .Case<VPInstruction>([&](VPInstruction *VPI) -> VPRecipeBase * {
        if (VPI->getOpcode() == VPInstruction::FirstOrderRecurrenceSplice) {
          assert(PrevEVL && "Fixed-order recurrences require previous EVL");
          VPValue *MinusOneVPV = VPI->getParent()->getPlan()->getOrAddLiveIn(
              ConstantInt::getSigned(Type::getInt32Ty(TypeInfo.getContext()),
                                     -1));
          SmallVector<VPValue *> Ops(VPI->operands());
          Ops.append({MinusOneVPV, &AllOneMask, PrevEVL, &EVL});
          return new VPWidenIntrinsicRecipe(Intrinsic::experimental_vp_splice,
                                            Ops, TypeInfo.inferScalarType(VPI),
                                            VPI->getDebugLoc());
        }

        VPValue *LHS, *RHS;
        // Transform select with a header mask condition
        //   select(header_mask, LHS, RHS)
        // into vector predication merge.
        //   vp.merge(all-true, LHS, RHS, EVL)
        if (!match(VPI, m_Select(m_Specific(HeaderMask), m_VPValue(LHS),
                                 m_VPValue(RHS))))
          return nullptr;
        // Use all true as the condition because this transformation is
        // limited to selects whose condition is a header mask.
        return new VPWidenIntrinsicRecipe(
            Intrinsic::vp_merge, {&AllOneMask, LHS, RHS, &EVL},
            TypeInfo.inferScalarType(LHS), VPI->getDebugLoc());
      })
      .Default([&](VPRecipeBase *R) { return nullptr; });
}

/// Replace recipes with their EVL variants.
static void transformRecipestoEVLRecipes(VPlan &Plan, VPValue &EVL) {
  Type *CanonicalIVType = Plan.getCanonicalIV()->getScalarType();
  VPTypeAnalysis TypeInfo(CanonicalIVType);
  LLVMContext &Ctx = CanonicalIVType->getContext();
  VPValue *AllOneMask = Plan.getOrAddLiveIn(ConstantInt::getTrue(Ctx));
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  VPBasicBlock *Header = LoopRegion->getEntryBasicBlock();

  // Create a scalar phi to track the previous EVL if fixed-order recurrence is
  // contained.
  VPScalarPHIRecipe *PrevEVL = nullptr;
  bool ContainsFORs =
      any_of(Header->phis(), IsaPred<VPFirstOrderRecurrencePHIRecipe>);
  if (ContainsFORs) {
    // TODO: Use VPInstruction::ExplicitVectorLength to get maximum EVL.
    VPValue *MaxEVL = &Plan.getVF();
    // Emit VPScalarCastRecipe in preheader if VF is not a 32 bits integer.
    if (unsigned VFSize =
            TypeInfo.inferScalarType(MaxEVL)->getScalarSizeInBits();
        VFSize != 32) {
      VPBuilder Builder(LoopRegion->getPreheaderVPBB());
      MaxEVL = Builder.createScalarCast(
          VFSize > 32 ? Instruction::Trunc : Instruction::ZExt, MaxEVL,
          Type::getInt32Ty(Ctx), DebugLoc());
    }
    PrevEVL = new VPScalarPHIRecipe(MaxEVL, &EVL, DebugLoc(), "prev.evl");
    PrevEVL->insertBefore(*Header, Header->getFirstNonPhi());
  }

  for (VPUser *U : to_vector(Plan.getVF().users())) {
    if (auto *R = dyn_cast<VPReverseVectorPointerRecipe>(U))
      R->setOperand(1, &EVL);
  }

  SmallVector<VPRecipeBase *> ToErase;

  for (VPValue *HeaderMask : collectAllHeaderMasks(Plan)) {
    for (VPUser *U : collectUsersRecursively(HeaderMask)) {
      auto *CurRecipe = cast<VPRecipeBase>(U);
      VPRecipeBase *EVLRecipe = createEVLRecipe(
          HeaderMask, *CurRecipe, TypeInfo, *AllOneMask, EVL, PrevEVL);
      if (!EVLRecipe)
        continue;

      [[maybe_unused]] unsigned NumDefVal = EVLRecipe->getNumDefinedValues();
      assert(NumDefVal == CurRecipe->getNumDefinedValues() &&
             "New recipe must define the same number of values as the "
             "original.");
      assert(
          NumDefVal <= 1 &&
          "Only supports recipes with a single definition or without users.");
      EVLRecipe->insertBefore(CurRecipe);
      if (isa<VPSingleDefRecipe, VPWidenLoadEVLRecipe>(EVLRecipe)) {
        VPValue *CurVPV = CurRecipe->getVPSingleValue();
        CurVPV->replaceAllUsesWith(EVLRecipe->getVPSingleValue());
      }
      // Defer erasing recipes till the end so that we don't invalidate the
      // VPTypeAnalysis cache.
      ToErase.push_back(CurRecipe);
    }
  }

  for (VPRecipeBase *R : reverse(ToErase)) {
    SmallVector<VPValue *> PossiblyDead(R->operands());
    R->eraseFromParent();
    for (VPValue *Op : PossiblyDead)
      recursivelyDeleteDeadRecipes(Op);
  }
}

/// Add a VPEVLBasedIVPHIRecipe and related recipes to \p Plan and
/// replaces all uses except the canonical IV increment of
/// VPCanonicalIVPHIRecipe with a VPEVLBasedIVPHIRecipe. VPCanonicalIVPHIRecipe
/// is used only for loop iterations counting after this transformation.
///
/// The function uses the following definitions:
///  %StartV is the canonical induction start value.
///
/// The function adds the following recipes:
///
/// vector.ph:
/// ...
///
/// vector.body:
/// ...
/// %EVLPhi = EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI [ %StartV, %vector.ph ],
///                                               [ %NextEVLIV, %vector.body ]
/// %AVL = sub original TC, %EVLPhi
/// %VPEVL = EXPLICIT-VECTOR-LENGTH %AVL
/// ...
/// %NextEVLIV = add IVSize (cast i32 %VPEVVL to IVSize), %EVLPhi
/// ...
///
/// If MaxSafeElements is provided, the function adds the following recipes:
/// vector.ph:
/// ...
///
/// vector.body:
/// ...
/// %EVLPhi = EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI [ %StartV, %vector.ph ],
///                                               [ %NextEVLIV, %vector.body ]
/// %AVL = sub original TC, %EVLPhi
/// %cmp = cmp ult %AVL, MaxSafeElements
/// %SAFE_AVL = select %cmp, %AVL, MaxSafeElements
/// %VPEVL = EXPLICIT-VECTOR-LENGTH %SAFE_AVL
/// ...
/// %NextEVLIV = add IVSize (cast i32 %VPEVL to IVSize), %EVLPhi
/// ...
///
bool VPlanTransforms::tryAddExplicitVectorLength(
    VPlan &Plan, const std::optional<unsigned> &MaxSafeElements) {
  VPBasicBlock *Header = Plan.getVectorLoopRegion()->getEntryBasicBlock();
  // The transform updates all users of inductions to work based on EVL, instead
  // of the VF directly. At the moment, widened inductions cannot be updated, so
  // bail out if the plan contains any.
  bool ContainsWidenInductions = any_of(
      Header->phis(),
      IsaPred<VPWidenIntOrFpInductionRecipe, VPWidenPointerInductionRecipe>);
  if (ContainsWidenInductions)
    return false;

  auto *CanonicalIVPHI = Plan.getCanonicalIV();
  VPValue *StartV = CanonicalIVPHI->getStartValue();

  // Create the ExplicitVectorLengthPhi recipe in the main loop.
  auto *EVLPhi = new VPEVLBasedIVPHIRecipe(StartV, DebugLoc());
  EVLPhi->insertAfter(CanonicalIVPHI);
  VPBuilder Builder(Header, Header->getFirstNonPhi());
  // Compute original TC - IV as the AVL (application vector length).
  VPValue *AVL = Builder.createNaryOp(
      Instruction::Sub, {Plan.getTripCount(), EVLPhi}, DebugLoc(), "avl");
  if (MaxSafeElements) {
    // Support for MaxSafeDist for correct loop emission.
    VPValue *AVLSafe = Plan.getOrAddLiveIn(
        ConstantInt::get(CanonicalIVPHI->getScalarType(), *MaxSafeElements));
    VPValue *Cmp = Builder.createICmp(ICmpInst::ICMP_ULT, AVL, AVLSafe);
    AVL = Builder.createSelect(Cmp, AVL, AVLSafe, DebugLoc(), "safe_avl");
  }
  auto *VPEVL = Builder.createNaryOp(VPInstruction::ExplicitVectorLength, AVL,
                                     DebugLoc());

  auto *CanonicalIVIncrement =
      cast<VPInstruction>(CanonicalIVPHI->getBackedgeValue());
  Builder.setInsertPoint(CanonicalIVIncrement);
  VPSingleDefRecipe *OpVPEVL = VPEVL;
  if (unsigned IVSize = CanonicalIVPHI->getScalarType()->getScalarSizeInBits();
      IVSize != 32) {
    OpVPEVL = Builder.createScalarCast(
        IVSize < 32 ? Instruction::Trunc : Instruction::ZExt, OpVPEVL,
        CanonicalIVPHI->getScalarType(), CanonicalIVIncrement->getDebugLoc());
  }
  auto *NextEVLIV = Builder.createOverflowingOp(
      Instruction::Add, {OpVPEVL, EVLPhi},
      {CanonicalIVIncrement->hasNoUnsignedWrap(),
       CanonicalIVIncrement->hasNoSignedWrap()},
      CanonicalIVIncrement->getDebugLoc(), "index.evl.next");
  EVLPhi->addOperand(NextEVLIV);

  transformRecipestoEVLRecipes(Plan, *VPEVL);

  // Replace all uses of VPCanonicalIVPHIRecipe by
  // VPEVLBasedIVPHIRecipe except for the canonical IV increment.
  CanonicalIVPHI->replaceAllUsesWith(EVLPhi);
  CanonicalIVIncrement->setOperand(0, CanonicalIVPHI);
  // TODO: support unroll factor > 1.
  Plan.setUF(1);
  return true;
}

void VPlanTransforms::dropPoisonGeneratingRecipes(
    VPlan &Plan,
    const std::function<bool(BasicBlock *)> &BlockNeedsPredication) {
  // Collect recipes in the backward slice of `Root` that may generate a poison
  // value that is used after vectorization.
  SmallPtrSet<VPRecipeBase *, 16> Visited;
  auto CollectPoisonGeneratingInstrsInBackwardSlice([&](VPRecipeBase *Root) {
    SmallVector<VPRecipeBase *, 16> Worklist;
    Worklist.push_back(Root);

    // Traverse the backward slice of Root through its use-def chain.
    while (!Worklist.empty()) {
      VPRecipeBase *CurRec = Worklist.pop_back_val();

      if (!Visited.insert(CurRec).second)
        continue;

      // Prune search if we find another recipe generating a widen memory
      // instruction. Widen memory instructions involved in address computation
      // will lead to gather/scatter instructions, which don't need to be
      // handled.
      if (isa<VPWidenMemoryRecipe, VPInterleaveRecipe, VPScalarIVStepsRecipe,
              VPHeaderPHIRecipe>(CurRec))
        continue;

      // This recipe contributes to the address computation of a widen
      // load/store. If the underlying instruction has poison-generating flags,
      // drop them directly.
      if (auto *RecWithFlags = dyn_cast<VPRecipeWithIRFlags>(CurRec)) {
        VPValue *A, *B;
        using namespace llvm::VPlanPatternMatch;
        // Dropping disjoint from an OR may yield incorrect results, as some
        // analysis may have converted it to an Add implicitly (e.g. SCEV used
        // for dependence analysis). Instead, replace it with an equivalent Add.
        // This is possible as all users of the disjoint OR only access lanes
        // where the operands are disjoint or poison otherwise.
        if (match(RecWithFlags, m_BinaryOr(m_VPValue(A), m_VPValue(B))) &&
            RecWithFlags->isDisjoint()) {
          VPBuilder Builder(RecWithFlags);
          VPInstruction *New = Builder.createOverflowingOp(
              Instruction::Add, {A, B}, {false, false},
              RecWithFlags->getDebugLoc());
          New->setUnderlyingValue(RecWithFlags->getUnderlyingValue());
          RecWithFlags->replaceAllUsesWith(New);
          RecWithFlags->eraseFromParent();
          CurRec = New;
        } else
          RecWithFlags->dropPoisonGeneratingFlags();
      } else {
        Instruction *Instr = dyn_cast_or_null<Instruction>(
            CurRec->getVPSingleValue()->getUnderlyingValue());
        (void)Instr;
        assert((!Instr || !Instr->hasPoisonGeneratingFlags()) &&
               "found instruction with poison generating flags not covered by "
               "VPRecipeWithIRFlags");
      }

      // Add new definitions to the worklist.
      for (VPValue *Operand : CurRec->operands())
        if (VPRecipeBase *OpDef = Operand->getDefiningRecipe())
          Worklist.push_back(OpDef);
    }
  });

  // Traverse all the recipes in the VPlan and collect the poison-generating
  // recipes in the backward slice starting at the address of a VPWidenRecipe or
  // VPInterleaveRecipe.
  auto Iter = vp_depth_first_deep(Plan.getEntry());
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(Iter)) {
    for (VPRecipeBase &Recipe : *VPBB) {
      if (auto *WidenRec = dyn_cast<VPWidenMemoryRecipe>(&Recipe)) {
        Instruction &UnderlyingInstr = WidenRec->getIngredient();
        VPRecipeBase *AddrDef = WidenRec->getAddr()->getDefiningRecipe();
        if (AddrDef && WidenRec->isConsecutive() &&
            BlockNeedsPredication(UnderlyingInstr.getParent()))
          CollectPoisonGeneratingInstrsInBackwardSlice(AddrDef);
      } else if (auto *InterleaveRec = dyn_cast<VPInterleaveRecipe>(&Recipe)) {
        VPRecipeBase *AddrDef = InterleaveRec->getAddr()->getDefiningRecipe();
        if (AddrDef) {
          // Check if any member of the interleave group needs predication.
          const InterleaveGroup<Instruction> *InterGroup =
              InterleaveRec->getInterleaveGroup();
          bool NeedPredication = false;
          for (int I = 0, NumMembers = InterGroup->getNumMembers();
               I < NumMembers; ++I) {
            Instruction *Member = InterGroup->getMember(I);
            if (Member)
              NeedPredication |= BlockNeedsPredication(Member->getParent());
          }

          if (NeedPredication)
            CollectPoisonGeneratingInstrsInBackwardSlice(AddrDef);
        }
      }
    }
  }
}

void VPlanTransforms::createInterleaveGroups(
    VPlan &Plan,
    const SmallPtrSetImpl<const InterleaveGroup<Instruction> *>
        &InterleaveGroups,
    VPRecipeBuilder &RecipeBuilder, const bool &ScalarEpilogueAllowed) {
  if (InterleaveGroups.empty())
    return;

  // Interleave memory: for each Interleave Group we marked earlier as relevant
  // for this VPlan, replace the Recipes widening its memory instructions with a
  // single VPInterleaveRecipe at its insertion point.
  VPDominatorTree VPDT;
  VPDT.recalculate(Plan);
  for (const auto *IG : InterleaveGroups) {
    SmallVector<VPValue *, 4> StoredValues;
    for (unsigned i = 0; i < IG->getFactor(); ++i)
      if (auto *SI = dyn_cast_or_null<StoreInst>(IG->getMember(i))) {
        auto *StoreR = cast<VPWidenStoreRecipe>(RecipeBuilder.getRecipe(SI));
        StoredValues.push_back(StoreR->getStoredValue());
      }

    bool NeedsMaskForGaps =
        IG->requiresScalarEpilogue() && !ScalarEpilogueAllowed;

    Instruction *IRInsertPos = IG->getInsertPos();
    auto *InsertPos =
        cast<VPWidenMemoryRecipe>(RecipeBuilder.getRecipe(IRInsertPos));

    // Get or create the start address for the interleave group.
    auto *Start =
        cast<VPWidenMemoryRecipe>(RecipeBuilder.getRecipe(IG->getMember(0)));
    VPValue *Addr = Start->getAddr();
    VPRecipeBase *AddrDef = Addr->getDefiningRecipe();
    if (AddrDef && !VPDT.properlyDominates(AddrDef, InsertPos)) {
      // TODO: Hoist Addr's defining recipe (and any operands as needed) to
      // InsertPos or sink loads above zero members to join it.
      bool InBounds = false;
      if (auto *Gep = dyn_cast<GetElementPtrInst>(
              getLoadStorePointerOperand(IRInsertPos)->stripPointerCasts()))
        InBounds = Gep->isInBounds();

      // We cannot re-use the address of member zero because it does not
      // dominate the insert position. Instead, use the address of the insert
      // position and create a PtrAdd adjusting it to the address of member
      // zero.
      assert(IG->getIndex(IRInsertPos) != 0 &&
             "index of insert position shouldn't be zero");
      auto &DL = IRInsertPos->getDataLayout();
      APInt Offset(32,
                   DL.getTypeAllocSize(getLoadStoreType(IRInsertPos)) *
                       IG->getIndex(IRInsertPos),
                   /*IsSigned=*/true);
      VPValue *OffsetVPV = Plan.getOrAddLiveIn(
          ConstantInt::get(IRInsertPos->getParent()->getContext(), -Offset));
      VPBuilder B(InsertPos);
      Addr = InBounds ? B.createInBoundsPtrAdd(InsertPos->getAddr(), OffsetVPV)
                      : B.createPtrAdd(InsertPos->getAddr(), OffsetVPV);
    }
    auto *VPIG = new VPInterleaveRecipe(IG, Addr, StoredValues,
                                        InsertPos->getMask(), NeedsMaskForGaps);
    VPIG->insertBefore(InsertPos);

    unsigned J = 0;
    for (unsigned i = 0; i < IG->getFactor(); ++i)
      if (Instruction *Member = IG->getMember(i)) {
        VPRecipeBase *MemberR = RecipeBuilder.getRecipe(Member);
        if (!Member->getType()->isVoidTy()) {
          VPValue *OriginalV = MemberR->getVPSingleValue();
          OriginalV->replaceAllUsesWith(VPIG->getVPValue(J));
          J++;
        }
        MemberR->eraseFromParent();
      }
  }
}

void VPlanTransforms::convertToConcreteRecipes(VPlan &Plan) {
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getEntry()))) {
    for (VPRecipeBase &R : make_early_inc_range(VPBB->phis())) {
      if (!isa<VPCanonicalIVPHIRecipe, VPEVLBasedIVPHIRecipe>(&R))
        continue;
      auto *PhiR = cast<VPHeaderPHIRecipe>(&R);
      StringRef Name =
          isa<VPCanonicalIVPHIRecipe>(PhiR) ? "index" : "evl.based.iv";
      auto *ScalarR =
          new VPScalarPHIRecipe(PhiR->getStartValue(), PhiR->getBackedgeValue(),
                                PhiR->getDebugLoc(), Name);
      ScalarR->insertBefore(PhiR);
      PhiR->replaceAllUsesWith(ScalarR);
      PhiR->eraseFromParent();
    }
  }
}

void VPlanTransforms::handleUncountableEarlyExit(
    VPlan &Plan, ScalarEvolution &SE, Loop *OrigLoop,
    BasicBlock *UncountableExitingBlock, VPRecipeBuilder &RecipeBuilder) {
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  auto *LatchVPBB = cast<VPBasicBlock>(LoopRegion->getExiting());
  VPBuilder Builder(LatchVPBB->getTerminator());
  auto *MiddleVPBB = Plan.getMiddleBlock();
  VPValue *IsEarlyExitTaken = nullptr;

  // Process the uncountable exiting block. Update IsEarlyExitTaken, which
  // tracks if the uncountable early exit has been taken. Also split the middle
  // block and have it conditionally branch to the early exit block if
  // EarlyExitTaken.
  auto *EarlyExitingBranch =
      cast<BranchInst>(UncountableExitingBlock->getTerminator());
  BasicBlock *TrueSucc = EarlyExitingBranch->getSuccessor(0);
  BasicBlock *FalseSucc = EarlyExitingBranch->getSuccessor(1);
  BasicBlock *EarlyExitIRBB =
      !OrigLoop->contains(TrueSucc) ? TrueSucc : FalseSucc;
  VPIRBasicBlock *VPEarlyExitBlock = Plan.getExitBlock(EarlyExitIRBB);

  VPValue *EarlyExitNotTakenCond = RecipeBuilder.getBlockInMask(
      OrigLoop->contains(TrueSucc) ? TrueSucc : FalseSucc);
  auto *EarlyExitTakenCond = Builder.createNot(EarlyExitNotTakenCond);
  IsEarlyExitTaken =
      Builder.createNaryOp(VPInstruction::AnyOf, {EarlyExitTakenCond});

  VPBasicBlock *NewMiddle = Plan.createVPBasicBlock("middle.split");
  VPBasicBlock *VectorEarlyExitVPBB =
      Plan.createVPBasicBlock("vector.early.exit");
  VPBlockUtils::insertOnEdge(LoopRegion, MiddleVPBB, NewMiddle);
  VPBlockUtils::connectBlocks(NewMiddle, VectorEarlyExitVPBB);
  NewMiddle->swapSuccessors();

  VPBlockUtils::connectBlocks(VectorEarlyExitVPBB, VPEarlyExitBlock);

  // Update the exit phis in the early exit block.
  VPBuilder MiddleBuilder(NewMiddle);
  VPBuilder EarlyExitB(VectorEarlyExitVPBB);
  for (VPRecipeBase &R : *VPEarlyExitBlock) {
    auto *ExitIRI = cast<VPIRInstruction>(&R);
    auto *ExitPhi = dyn_cast<PHINode>(&ExitIRI->getInstruction());
    if (!ExitPhi)
      break;

    VPValue *IncomingFromEarlyExit = RecipeBuilder.getVPValueOrAddLiveIn(
        ExitPhi->getIncomingValueForBlock(UncountableExitingBlock));

    if (OrigLoop->getUniqueExitBlock()) {
      // If there's a unique exit block, VPEarlyExitBlock has 2 predecessors
      // (MiddleVPBB and NewMiddle). Add the incoming value from MiddleVPBB
      // which is coming from the original latch.
      VPValue *IncomingFromLatch = RecipeBuilder.getVPValueOrAddLiveIn(
          ExitPhi->getIncomingValueForBlock(OrigLoop->getLoopLatch()));
      ExitIRI->addOperand(IncomingFromLatch);
      ExitIRI->extractLastLaneOfOperand(MiddleBuilder);
    }
    // Add the incoming value from the early exit.
    if (!IncomingFromEarlyExit->isLiveIn())
      IncomingFromEarlyExit =
          EarlyExitB.createNaryOp(VPInstruction::ExtractFirstActive,
                                  {IncomingFromEarlyExit, EarlyExitTakenCond});
    ExitIRI->addOperand(IncomingFromEarlyExit);
  }
  MiddleBuilder.createNaryOp(VPInstruction::BranchOnCond, {IsEarlyExitTaken});

  // Replace the condition controlling the non-early exit from the vector loop
  // with one exiting if either the original condition of the vector latch is
  // true or the early exit has been taken.
  auto *LatchExitingBranch = cast<VPInstruction>(LatchVPBB->getTerminator());
  assert(LatchExitingBranch->getOpcode() == VPInstruction::BranchOnCount &&
         "Unexpected terminator");
  auto *IsLatchExitTaken =
      Builder.createICmp(CmpInst::ICMP_EQ, LatchExitingBranch->getOperand(0),
                         LatchExitingBranch->getOperand(1));
  auto *AnyExitTaken = Builder.createNaryOp(
      Instruction::Or, {IsEarlyExitTaken, IsLatchExitTaken});
  Builder.createNaryOp(VPInstruction::BranchOnCond, AnyExitTaken);
  LatchExitingBranch->eraseFromParent();
}

void VPlanTransforms::materializeLiveInBroadcasts(VPlan &Plan) {
  if (Plan.hasScalarVFOnly())
    return;

  VPDominatorTree VPDT;
  VPDT.recalculate(Plan);
  auto *VectorPreheader = Plan.getVectorPreheader();
  VPBuilder Builder(VectorPreheader);
  for (VPValue *LiveIn : Plan.getLiveIns()) {
    if (all_of(LiveIn->users(),
               [LiveIn](VPUser *U) {
                 return cast<VPRecipeBase>(U)->usesScalars(LiveIn);
               }) ||
        !LiveIn->getLiveInIRValue() ||
        isa<Constant>(LiveIn->getLiveInIRValue()))
      continue;

    // Add explicit broadcast if the vector preheader dominates all users.
    // TODO: Find valid insert point for all users.
    if (all_of(LiveIn->users(), [&VPDT, VectorPreheader](VPUser *U) {
          return VectorPreheader != cast<VPRecipeBase>(U)->getParent() &&
                 VPDT.dominates(VectorPreheader,
                                cast<VPRecipeBase>(U)->getParent());
        })) {
      auto *Broadcast =
          Builder.createNaryOp(VPInstruction::Broadcast, {LiveIn});
      LiveIn->replaceUsesWithIf(Broadcast, [LiveIn, Broadcast](VPUser &U,
                                                               unsigned Idx) {
        return Broadcast != &U && !cast<VPRecipeBase>(&U)->usesScalars(LiveIn);
      });
    }
  }
}

// Given a loop with multiple exits (\p Exiting), change the CFG so that
// there is a single exiting block, and return that block.
static VPBasicBlock *createSingleExitLoop(VPlan &Plan, VPDominatorTree &DT,
                                          VPBasicBlock *Header,
                                          ArrayRef<VPBlockBase *> Exiting,
                                          VPBasicBlock *OrigLatch,
                                          VPBlockBase *Exit) {
  assert(is_contained(Exiting, OrigLatch));
  SmallVector<VPBlockBase *> OrigExitPredecessors(Exit->predecessors());

  // Create a new latch and make all early exits a branch into that new latch.
  // Also create a PHI that will be used as the new exit condition.
  auto &Ctx = Plan.getCanonicalIV()->getScalarType()->getContext();
  VPBasicBlock *NewLatch = Plan.createVPBasicBlock(OrigLatch->getName());
  auto *NewExitCond = new VPWidenPHIRecipe(nullptr);
  SmallVector<cfg::Update<VPBlockBase *>> DTUs;
  for (VPBlockBase *BB : Exiting) {
    auto *Term = cast<VPInstruction>(cast<VPBasicBlock>(BB)->getTerminator());
    if (Term->getOpcode() != VPInstruction::BranchOnCond)
      return nullptr;

    VPValue *PhiIncVal = Plan.getOrAddLiveIn(
        ConstantInt::getBool(Ctx, BB->getSuccessors()[0] == Exit));
    VPBlockUtils::replaceSuccessor(Exit, NewLatch, BB);
    NewExitCond->addOperand(PhiIncVal);
    DTUs.append({{VPDominatorTree::Insert, BB, NewLatch},
                 {VPDominatorTree::Delete, BB, Exit}});
  }
  VPBlockUtils::connectBlocks(NewLatch, Exit);
  VPBlockUtils::replacePredecessor(OrigLatch, NewLatch, Header);
  DTUs.append({{VPDominatorTree::Insert, NewLatch, Exit},
               {VPDominatorTree::Insert, NewLatch, Header},
               {VPDominatorTree::Delete, OrigLatch, Header}});
  DT.applyUpdates(DTUs);
  assert(DT.verify(VPDominatorTree::VerificationLevel::Fast));
  NewLatch->appendRecipe(NewExitCond);
  NewLatch->appendRecipe(
      new VPInstruction(VPInstruction::BranchOnCond, {NewExitCond}));

  // Repair the loop-header phi's incoming values: If defined "after" a early
  // exit, they might not dominate the latch anymore, so a second phi needs
  // to be created in the new latch.
  for (VPRecipeBase &R : Header->phis()) {
    auto *Phi = cast<VPWidenPHIRecipe>(&R);
    unsigned PhiBackedgeIdx = Phi->getIncomingBlock(0) != NewLatch;
    auto *BackedgeDef =
        Phi->getIncomingValue(PhiBackedgeIdx)->getDefiningRecipe();
    if (!BackedgeDef || DT.dominates(BackedgeDef->getParent(), NewLatch))
      continue;

    auto *IRPhi = cast<PHINode>(Phi->getUnderlyingValue());
    auto *NewPhi = new VPWidenPHIRecipe(IRPhi);
    NewPhi->insertBefore(*NewLatch, NewLatch->getFirstNonPhi());
    Phi->setOperand(PhiBackedgeIdx, NewPhi);
    VPValue *Poison = Plan.getOrAddLiveIn(PoisonValue::get(IRPhi->getType()));
    for (VPBlockBase *Pred : NewLatch->predecessors())
      if (Pred == OrigLatch)
        NewPhi->addOperand(BackedgeDef->getVPSingleValue());
      else
        NewPhi->addOperand(Poison);
  }

  // Repair the LCSSA phis in the exit block.
  for (VPRecipeBase &R : cast<VPBasicBlock>(Exit)->phis()) {
    auto *OrigPhi = cast<VPWidenPHIRecipe>(&R);
    assert(OrigPhi->getNumOperands() == OrigExitPredecessors.size() &&
           Exit->getNumPredecessors() == 1);

    // Create a new phi in the new latch that will forward the values
    // that previously exited the loop.
    auto *IRPhi = cast<PHINode>(OrigPhi->getUnderlyingValue());
    auto *NewLatchPhi = new VPWidenPHIRecipe(IRPhi);
    NewLatchPhi->insertBefore(*NewLatch, NewLatch->getFirstNonPhi());
    for (VPBlockBase *Pred : NewLatch->predecessors()) {
      auto Idx = std::distance(OrigExitPredecessors.begin(),
                               find(OrigExitPredecessors, Pred));
      NewLatchPhi->addOperand(OrigPhi->getOperand(Idx));
    }

    // Replace the old LCSSA phi by a new one.
    auto *NewLCSSAPhi = new VPWidenPHIRecipe(IRPhi);
    NewLCSSAPhi->addOperand(NewLatchPhi);
    OrigPhi->replaceAllUsesWith(NewLCSSAPhi);
    OrigPhi->eraseFromParent();
  }

  return NewLatch;
}

// Given a VPlan that potentially contains inner loops (inside the assumed to
// already exist main vector region), create VPRegionBlocks for these loops.
static bool detectInnerLoopsAndAddRegions(VPlan &Plan, VPDominatorTree &DT) {
  auto *VectorLoopRegion = Plan.getVectorLoopRegion();
  assert(VectorLoopRegion && "Outer-most loop region should already exist");

  // A map of a loop header to a list of latches.
  DenseMap<VPBasicBlock *, SmallVector<VPBasicBlock *>> Latches;
  for (VPBasicBlock *BB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getEntry())))
    // Look for edges from a block to one that dominates that block.
    for (VPBasicBlock *Succ :
         VPBlockUtils::blocksOnly<VPBasicBlock>(BB->successors()))
      if (DT.dominates(Succ, BB))
        Latches[Succ].push_back(BB);

  // Visit all detected loop headers and create regions for those.
  for (auto &[Header, Latches] : Latches) {
    SmallSetVector<VPBlockBase *, 1> ExitingBlocks;
    SmallSetVector<VPBlockBase *, 1> ExitBlocks;

    // Build the set of blocks in the loop by going up from the latches and
    // adding all blocks dominated by the header.
    SmallSetVector<VPBlockBase *, 4> Blocks;
    Blocks.insert(Header);
    Blocks.insert(Latches.begin(), Latches.end());
    for (unsigned I = 0; I < Blocks.size(); ++I) {
      auto *BB = Blocks[I];
      for (auto *Pred : BB->predecessors())
        Blocks.insert(Pred);
    }

    // Find exiting blocks:
    for (VPBlockBase *BB : Blocks)
      for (auto *Succ : BB->successors())
        if (!DT.dominates(Header, Succ)) {
          ExitingBlocks.insert(BB);
          ExitBlocks.insert(Succ);
        }

    if (Latches.size() != 1 || Header->getNumPredecessors() != 2 ||
        ExitBlocks.size() != 1 || !is_contained(ExitingBlocks, Latches[0]) ||
        ExitBlocks[0]->getNumPredecessors() != ExitingBlocks.size())
      return false; // Expected inner loops exiting through a latch with
                    // a single latch and a single (dedicated) exit block.

    VPBasicBlock *Latch = Latches[0];
    VPBlockBase *Exit = ExitBlocks[0];
    if (ExitingBlocks.size() > 1) {
      Latch = createSingleExitLoop(Plan, DT, Header,
                                   ExitingBlocks.getArrayRef(), Latch, Exit);
      if (!Latch)
        return false; // The transformation to a single exit did not work.
    }

    // Create the region and insert it into the VPlan.
    bool ExitIfTrue = Header == Latch->getSuccessors()[1];
    VPBlockBase::VPBlocksTy::iterator Preheader =
        find_if(Header->predecessors(),
                [&](VPBlockBase *Pred) { return !Blocks.contains(Pred); });
    assert(Preheader != Header->predecessors().end());
    bool PreheaderIsFirstPred = *Preheader == Header->getPredecessors()[0];
    VPRegionBlock *Region =
        Plan.createVPRegionBlock(Header->getName(), /*IsReplicator*/ false);
    Region->setParent(Header->getParent());
    VPBlockUtils::disconnectBlocks(*Preheader, Header);
    VPBlockUtils::connectBlocks(*Preheader, Region);
    VPBlockUtils::disconnectBlocks(Latch, Header);
    VPBlockUtils::disconnectBlocks(Latch, Exit);
    VPBlockUtils::connectBlocks(Region, Exit);
    Region->setEntry(Header);
    Region->setExiting(Latch);
    for (VPBlockBase *B : Blocks)
      B->setParent(Region);

    // Ensure that the inner region is always exited when the condition is true.
    if (!ExitIfTrue) {
      auto *Br = cast<VPInstruction>(&Latch->back());
      assert(Br->getOpcode() == VPInstruction::BranchOnCond);
      auto *Not = new VPInstruction(VPInstruction::Not, {Br->getOperand(0)},
                                    Br->getDebugLoc());
      Not->insertBefore(Br);
      Br->setOperand(0, Not);
    }

    // Make sure loop-header phis have the preheader as first operand and the
    // backedge value as second operand.
    if (!PreheaderIsFirstPred)
      for (VPRecipeBase &Phi : Header->phis()) {
        assert(isa<VPWidenPHIRecipe>(&Phi));
        VPValue *Op0 = Phi.getOperand(0);
        Phi.setOperand(0, Phi.getOperand(1));
        Phi.setOperand(1, Op0);
      }

    // Update the dominator tree.
    DT.applyUpdates({{VPDominatorTree::Insert, *Preheader, Region},
                     {VPDominatorTree::Insert, Region, Header},
                     {VPDominatorTree::Insert, Region, Exit},
                     {VPDominatorTree::Delete, *Preheader, Header},
                     {VPDominatorTree::Delete, Latch, Header},
                     {VPDominatorTree::Delete, Latch, Exit}});
  }

  assert(DT.verify(VPDominatorTree::VerificationLevel::Fast));
  return true;
}

// Given a block \p BB and two of its predecessors, make sure that there exists
// a block with only those two predecessors and no other ones. If such a block
// needs to be created return it (instead of BB) and connect it to the original
// common successor. This can unlock BOSCC branches or preservation of uniform
// branches.
static VPBasicBlock *createDedicatedJoinBlock(VPlan &Plan, VPDominatorTree &DT,
                                              VPBasicBlock *BB,
                                              VPBlockBase *LHSPred,
                                              VPBlockBase *RHSPred) {
  assert(BB->getNumPredecessors() >= 2 && LHSPred != RHSPred &&
         is_contained(BB->predecessors(), LHSPred) &&
         is_contained(BB->predecessors(), RHSPred) &&
         "Different inputs expected");
  if (BB->getNumPredecessors() == 2)
    return BB;

  auto *NewBB = Plan.createVPBasicBlock(
      Twine(LHSPred->getName()) + ".joins." + RHSPred->getName());

  // Modify the CFG and update the dominator tree.
  SmallVector<VPBlockBase *> OrigBBPreds(BB->predecessors());
  VPBlockUtils::replaceSuccessor(BB, NewBB, LHSPred);
  VPBlockUtils::replaceSuccessor(BB, NewBB, RHSPred);
  VPBlockUtils::connectBlocks(NewBB, BB);
  DT.applyUpdates({{VPDominatorTree::Insert, LHSPred, NewBB},
                   {VPDominatorTree::Insert, RHSPred, NewBB},
                   {VPDominatorTree::Insert, NewBB, BB},
                   {VPDominatorTree::Delete, LHSPred, BB},
                   {VPDominatorTree::Delete, RHSPred, BB}});

  // Fix phi nodes by creating new ones in NewBB and using
  // those as operands in BB instead of the values from LHSPred/RHSPred.
  for (VPRecipeBase &R : make_early_inc_range(BB->phis())) {
    auto *OrigPhi = cast<VPWidenPHIRecipe>(&R);
    auto *IRPhi = cast<PHINode>(OrigPhi->getUnderlyingValue());

    // Create the phi node for NewBB.
    auto *NewBBPhi = new VPWidenPHIRecipe(IRPhi);
    auto LHSIdx = std::distance(OrigBBPreds.begin(), find(OrigBBPreds, LHSPred));
    NewBBPhi->addOperand(OrigPhi->getOperand(LHSIdx));
    auto RHSIdx = std::distance(OrigBBPreds.begin(), find(OrigBBPreds, RHSPred));
    NewBBPhi->addOperand(OrigPhi->getOperand(RHSIdx));
    NewBB->appendRecipe(NewBBPhi);

    // Create a new phi node replacing OrigPhi.
    auto *NewPhi = new VPWidenPHIRecipe(IRPhi);
    NewPhi->insertAfter(OrigPhi);
    for (VPBlockBase *Pred : BB->predecessors())
      if (Pred == NewBB) {
        NewPhi->addOperand(NewBBPhi);
      } else {
        auto Idx = std::distance(OrigBBPreds.begin(), find(OrigBBPreds, Pred));
        NewPhi->addOperand(OrigPhi->getOperand(Idx));
      }
    OrigPhi->replaceAllUsesWith(NewPhi);
    OrigPhi->eraseFromParent();
  }

  return NewBB;
}

static std::optional<std::tuple<VPBasicBlock *, VPBlockBase *, VPBlockBase *>>
canKeepBranchDuringIfConversion(const VPDominatorTree &DT, VPBasicBlock *VPBB) {
  const VPRegionBlock *Region = VPBB->getParent();
  auto FindSubregionExit =
      [&](VPBasicBlock *Pred,
          VPBlockBase *Entry) -> std::pair<VPBlockBase *, VPBlockBase *> {
    // The branch preservation is restricted to cases where
    // the SESEs are completely empty or have a dedicated entry and exit.
    // Because of the way the VPlan is flattened, the entry could already
    // have gotten predecessors removed, so check based on the IR.
    if (Entry->getNumPredecessors() >= 2)
      return {Pred, Entry};

    // Build the biggest possible SESE with the entry Entry.
    // As the DT is not updated during flattening, even if other edges
    // entering the SESE would have already been removed, the fact
    // that there used to be one will be detected.
    VPBlockBase *Exiting = nullptr;
    SmallSetVector<VPBlockBase *, 4> Worklist;
    Worklist.insert(Entry);
    for (unsigned I = 0; I < Worklist.size(); I++) {
      auto *BB = Worklist[I];
      assert(BB->getParent() == Region);
      for (auto *Succ : BB->getSuccessors()) {
        if (DT.dominates(Entry, Succ))
          Worklist.insert(Succ);
        else if (Exiting || BB->getNumSuccessors() != 1)
          return {nullptr, nullptr};
        else
          Exiting = BB;
      }
    }

    return {Exiting, Exiting->getSingleSuccessor()};
  };

  auto [LHSExiting, LHSSucc] =
      FindSubregionExit(VPBB, VPBB->getSuccessors()[0]);
  auto [RHSExiting, RHSSucc] =
      FindSubregionExit(VPBB, VPBB->getSuccessors()[1]);
  if (!LHSExiting || !RHSExiting || LHSSucc != RHSSucc ||
      !isa<VPBasicBlock>(LHSSucc))
    return std::nullopt;

  return std::tuple(cast<VPBasicBlock>(LHSSucc), LHSExiting, RHSExiting);
}

std::optional<DenseMap<VPBlockBase *, VPValue *>>
VPlanTransforms::linarizeAndCollectMasks(
    VPlan &Plan, VPValue *HeaderMask,
    const std::function<bool(const BranchInst &)> &IsUniform) {
  using namespace VPlanPatternMatch;

  // Create VPRegionBlocks for inner loops.
  VPDominatorTree DT;
  DT.recalculate(Plan);
  if (!detectInnerLoopsAndAddRegions(Plan, DT))
    return std::nullopt;

  // Find uniform branches at the head of two single-entry single-exit
  // subregions that join at the same block so that the branch can be
  // preserved while linearizing/if-converting all others.
  VPRegionBlock *VectorLoopRegion = Plan.getVectorLoopRegion();
  VPBasicBlock *Header = VectorLoopRegion->getEntryBasicBlock();
  DenseMap<VPBlockBase *, VPBlockBase *> UniformBranch2Join, UniformJoin2Branch;
  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>> RPOT(
      Header);
  for (VPBasicBlock *BB : VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT)) {
    if (!BB->getParent())
      break; // Ignore the middle block and anything after it.
    if (BB->getNumSuccessors() > 2)
      return std::nullopt; // TODO: Support switch terminators.
    if (BB->getNumSuccessors() != 2 ||
        !IsUniform(*cast<BranchInst>(
            cast<VPInstruction>(BB->getTerminator())->getUnderlyingValue())))
      continue;
    auto JoinInfo = canKeepBranchDuringIfConversion(DT, BB);
    if (!JoinInfo)
      continue;

    auto [JoinBB, LHSExitingBB, RHSExitingBB] = JoinInfo.value();
    if (JoinBB->getNumPredecessors() > 2)
      JoinBB = createDedicatedJoinBlock(Plan, DT, JoinBB, LHSExitingBB,
                                        RHSExitingBB);
    UniformBranch2Join[BB] = JoinBB;
    UniformJoin2Branch[JoinBB] = BB;
  }

  VPBuilder Builder;
  DenseMap<std::pair<VPBlockBase *, VPBlockBase *>, VPValue *> EdgeMasks;
  DenseMap<VPBlockBase *, VPValue *> BlockMasks;
  BlockMasks[Header] = HeaderMask;

  auto SetMaskInsertPos = [&](VPBasicBlock *BB, VPBuilder &Builder) {
    // Prev. mask calculations will not have a underlying instr.,
    // so skip over those.
    VPBasicBlock::iterator IP = BB->getFirstNonPhi();
    while (IP != BB->end() && !IP->getVPSingleValue()->getUnderlyingValue())
      ++IP;
    Builder.setInsertPoint(BB, IP);
  };

  // Create a new mask based on the mask of Pred and the condition for the
  // edge from Pred to Succ.
  auto CreateEdgeMask = [&](VPBasicBlock *Pred,
                            VPBasicBlock *Succ) -> VPValue * {
    if (auto Iter = EdgeMasks.find({Pred, Succ}); Iter != EdgeMasks.end())
      return Iter->second;

    VPValue *PredMask = BlockMasks.at(Pred);
    assert(is_contained(Succ->predecessors(), Pred) &&
           is_contained(Pred->successors(), Succ));
    if (Pred->getNumSuccessors() == 1 ||
        (Pred->getNumSuccessors() == 2 &&
         Pred->getSuccessors()[0] == Pred->getSuccessors()[1])) {
      EdgeMasks[std::pair(Pred, Succ)] = PredMask;
      return PredMask;
    }

    auto *Term = cast<VPInstruction>(Pred->getTerminator());
    assert(Term->getOpcode() == VPInstruction::BranchOnCond &&
           "Predication of switch terminators unimplemented");
    SetMaskInsertPos(Succ, Builder);
    VPValue *Cond = Term->getOperand(0);
    if (Pred->getSuccessors()[1] == Succ)
      Cond = Builder.createNot(Cond, Term->getDebugLoc());

    auto *Mask = PredMask
                     ? Builder.createAnd(PredMask, Cond, Term->getDebugLoc())
                     : Cond;
    EdgeMasks[std::pair(Pred, Succ)] = Mask;
    return Mask;
  };

  // Based on the edges from predecessors to BB, create a new mask for
  // BB itself.
  auto CreateBlockMask = [&](VPBasicBlock *BB) -> VPValue * {
    if (auto Iter = BlockMasks.find(BB); Iter != BlockMasks.end())
      return Iter->second;

    VPValue *Mask = nullptr;
    SetMaskInsertPos(BB, Builder);
    for (VPBlockBase *Pred : BB->predecessors()) {
      VPValue *EdgeMask = EdgeMasks.at({Pred->getExitingBasicBlock(), BB});
      if (!EdgeMask) {
        BlockMasks[BB] = nullptr;
        return nullptr;
      }
      if (!Mask) {
        Mask = EdgeMask;
        continue;
      }

      Mask = Builder.createOr(Mask, EdgeMask);
    }

    VPValue *V = nullptr;
    if (Mask && match(Mask, m_c_BinaryOr(m_Not(m_VPValue(V)), m_Specific(V))))
      Mask = V;

    BlockMasks[BB] = Mask;
    return Mask;
  };

  // Handle inner loops, especially if they have non-uniform trip-counts.
  auto PredicateInnerLoop = [&](VPRegionBlock &Region) {
    // The mask for the exit block should always be that of the preheader.
    auto *Exit = cast<VPBasicBlock>(Region.getSingleSuccessor());
    auto *Preheader = Region.getSinglePredecessor();
    assert(Exit->getNumPredecessors() == 1 &&
           Preheader->getNumSuccessors() == 1 &&
           "Expected dedicated exits and preheaders");
    VPValue *PreheaderMask = BlockMasks.at(Preheader);
    BlockMasks[Exit] = PreheaderMask;

    // If the trip-count of the loop is uniform, then all lanes of the
    // vectoized loop will exit the inner loop at once, and the inner
    // loop can use the same mask as the preheader.
    auto *Entry = Region.getEntryBasicBlock();
    auto *Exiting = Region.getExitingBasicBlock();
    auto *Term = cast<VPInstruction>(Exiting->getTerminator());
    const BranchInst *IRBr = cast<BranchInst>(Term->getUnderlyingValue());
    if (IsUniform(*IRBr)) {
      BlockMasks[Entry] = PreheaderMask;
      return;
    }

    // In case the trip-count is not uniform, a active-lane mask is needed that
    // will be a PHI which starts with the preheader mask and where the backedge
    // value is true only for lanes that were active in the previous iteration
    // and where the exit condition did not become true.
    LLVMContext &Ctx = IRBr->getContext();
    if (!PreheaderMask)
      PreheaderMask = Plan.getOrAddLiveIn(ConstantInt::getTrue(Ctx));
    auto *ALM = new VPWidenPHIRecipe(nullptr);
    ALM->setIsActiveLaneMask(true);
    ALM->addOperand(PreheaderMask);
    ALM->insertBefore(*Entry, Entry->getFirstNonPhi());
    BlockMasks[Entry] = ALM;
    Builder.setInsertPoint(Term);
    DebugLoc DL(Term->getDebugLoc());
    auto *NextALM = Builder.createLogicalAnd(
        ALM, Builder.createNot(Term->getOperand(0), DL), DL);
    ALM->addOperand(NextALM);
    auto *Any = Builder.createNaryOp(VPInstruction::AnyOf, {NextALM}, DL);
    Term->setOperand(0, Builder.createNot(Any, DL));

    // Handle LCSSA phis and live-out values, which require a passthrough
    // PHI/select pair to make sure the last active value for each lane
    // leaves the loop.
    for (VPRecipeBase &R : Exit->phis()) {
      auto &LCSSAPhi = *cast<VPWidenPHIRecipe>(&R);
      assert(LCSSAPhi.getNumOperands() == 1);
      auto *OutVal = LCSSAPhi.getOperand(0);
      auto *Def = OutVal->getDefiningRecipe();
      if (!Def || Def->getParent()->getParent() != &Region)
        continue;

      VPWidenPHIRecipe *HeaderPhi = nullptr;
      // Check if there already is a PHI that uses this value as
      // backedge value, and if so, use that to generate the live-out.
      // NOTE: If this header phi has a uniform use which relies on
      // lane zero of the PHI beeing the latest value, than this
      // creates a temporarily semantically wrong VPlan.
      // This is then fixed in optimizeInnerInductions().
      for (VPRecipeBase &R : Entry->phis()) {
        auto *Phi = cast<VPWidenPHIRecipe>(&R);
        if (Phi->getOperand(1) == OutVal) {
          HeaderPhi = Phi;
          break;
        }
      }
      // If there is no such PHI, create a new one:
      if (!HeaderPhi) {
        HeaderPhi = new VPWidenPHIRecipe(nullptr);
        auto *Poison = Plan.getOrAddLiveIn(
            PoisonValue::get(LCSSAPhi.getUnderlyingValue()->getType()));
        HeaderPhi->addOperand(Poison);
        // Will be replaced by select:
        HeaderPhi->addOperand(Poison);
        HeaderPhi->insertBefore(*Entry, Entry->getFirstNonPhi());
      }
      auto *Select =
          new VPInstruction(Instruction::Select, {ALM, OutVal, HeaderPhi});
      Select->insertBefore(Term);
      HeaderPhi->setOperand(1, Select);
      LCSSAPhi.setOperand(0, Select);
    }
  };

  // Traverse the loop, predicate instructions, and linearize the control
  // flow if the branch cannot be preserved.
  assert(DT.verify(VPDominatorTree::VerificationLevel::Fast));
  VPBlockBase *RPOPred = nullptr;
  for (VPBasicBlock *BB : VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT)) {
    // Don't visit the tail loop or even the middle block.
    VPRegionBlock *Region = BB->getParent();
    if (!Region)
      break; // Ignore the middle block and anything after it.
    if (Region->getEntry() == BB && Region != VectorLoopRegion) {
      PredicateInnerLoop(*Region);
      VPBlockUtils::connectBlocks(RPOPred, Region);
      VPBlockUtils::disconnectBlocks(Region, Region->getSingleSuccessor());
      RPOPred = BB;
      continue;
    }

    // Create the mask before any blends.
    VPValue *Mask = CreateBlockMask(BB);

    // Keep loop-header, LCSSA, and uniform-branch-join phis, replace the
    // rest by blends.
    VPBlockBase *JoinBBOfBranch = UniformJoin2Branch.lookup(BB);
    if (!JoinBBOfBranch && !isa_and_nonnull<VPRegionBlock>(RPOPred) &&
        BB != Header) {
      SmallVector<VPValue *, 4> Ops;
      Ops.reserve(BB->getNumPredecessors() * 2);
      for (VPRecipeBase &R : make_early_inc_range(BB->phis())) {
        auto *Phi = cast<VPWidenPHIRecipe>(&R);
        assert(Phi->getNumOperands() == BB->getNumPredecessors());
        auto *IRPhi = cast<PHINode>(Phi->getUnderlyingValue());
        Ops.clear();
        if (BB->getNumPredecessors() == 1)
          // Just replacing all uses does not work because live-out uses
          // are not visible in the VPlan yet, and keeping a PHI results
          // in invalid VPlans after blocks are merged into predecessors.
          Ops.push_back(Phi->getOperand(0));
        else
          for (auto [Val, Pred] : zip(Phi->operands(), BB->predecessors()))
            Ops.append({Val, EdgeMasks.at({Pred, BB})});
        auto *Blend = new VPBlendRecipe(IRPhi, Ops);
        Builder.insert(Blend);
        Phi->replaceAllUsesWith(Blend);
        Phi->eraseFromParent();
      }
    }

    // A block that contains a preserved uniform branch or
    // that is exiting a subregion into the join block of a
    // preserved uniform branch keeps its original successors.
    bool KeepSuccs = UniformBranch2Join.contains(BB) ||
                     UniformJoin2Branch.contains(BB->getSingleSuccessor());
    // A block directly following a preserved uniform branch or
    // the block where the two subregions join back together
    // keeps its original predcessors.
    bool KeepPreds = UniformJoin2Branch.contains(BB) ||
                     UniformBranch2Join.contains(BB->getSinglePredecessor());

    // Pre-calculate any exiting edge masks.
    for (VPBlockBase *Succ : BB->successors())
      if (KeepSuccs)
        EdgeMasks[std::pair(BB, Succ)] = Mask;
      else if (auto *SuccBB = dyn_cast<VPBasicBlock>(Succ))
        CreateEdgeMask(BB, SuccBB);

    if (!KeepPreds)
      for (auto *Pred : to_vector(BB->predecessors()))
        VPBlockUtils::disconnectBlocks(Pred, BB);
    if (!KeepPreds && RPOPred)
      VPBlockUtils::connectBlocks(RPOPred, BB);

    // Remove the terminator and select the RPO pred. of the next BB.
    if (Region->getExiting() == BB) {
      RPOPred = Region;
    } else {
      if (!KeepSuccs && !BB->empty() &&
          match(&BB->back(), m_BranchOnCond(m_VPValue())))
        BB->back().eraseFromParent();
      RPOPred = BB;
    }
  }

  return BlockMasks;
}

// The min. estimated probability that the BOSCC branch will be taken.
// A higher probability makes BOSCC branch generation less likely.
static const BranchProbability MinBOSCCJumpOverProbability =
    BranchProbability::getBranchProbability(3, 4);

// Used to hoist masks out of a region for which a
// BOSCC branch is created. This can be necessary for the
// anyof check and avoid PHIs in the successor.
static void hoistMasksInFrontOf(VPValue *V, VPDominatorTree &DT,
                                VPBasicBlock &Dst, VPBasicBlock::iterator IP) {
  auto *Def = dyn_cast<VPInstruction>(V);
  if (!Def || !(Def->getOpcode() == VPInstruction::Not ||
                Def->getOpcode() == VPInstruction::LogicalAnd ||
                Def->getOpcode() == Instruction::Or))
    return;

  // If this function were to be generalized to any non-side-effecting
  // or memory-accessing instruction, then more care would need to be
  // taken so that operands stay in a topological order.
  // For masks, this will always be the case.
  for (VPValue *Op : Def->operands())
    hoistMasksInFrontOf(Op, DT, Dst, IP);
  Def->moveBefore(Dst, IP);
}

void VPlanTransforms::introduceBOSCCBranches(
    VPlan &Plan, VPDominatorTree &DT, ElementCount VF, unsigned IC,
    ArrayRef<std::tuple<VPValue *, VPBlockBase *, VPBlockBase *>> SESEs,
    VPCostContext &CostCtx) {
  assert(DT.verify(VPDominatorTree::VerificationLevel::Fast));
  assert(verifyVPlanIsValid(Plan));
  if (SESEs.empty())
    return;

  unsigned VFxIC = IC * VF.getKnownMinValue();
  if (VF.isScalable())
    VFxIC *= CostCtx.TTI.getVScaleForTuning().value_or(0);

  // If the cost of the region a BOSCC branch allows us to jump over
  // is lower than that of the check and branch introduced by BOSCC,
  // then don't do it.
  Type *Int1Ty = IntegerType::getInt1Ty(CostCtx.LLVMCtx);
  InstructionCost MinBOSCCCost =
      CostCtx.TTI.getArithmeticReductionCost(Instruction::Or,
                                             VectorType::get(Int1Ty, VF),
                                             std::nullopt, CostCtx.CostKind) *
          IC +
      CostCtx.TTI.getArithmeticInstrCost(Instruction::Or, Int1Ty,
                                         CostCtx.CostKind) *
          (IC - 1) +
      CostCtx.TTI.getCFInstrCost(Instruction::Br, CostCtx.CostKind) * 2;
  LLVM_DEBUG(dbgs() << "BOSCC: Min. region cost: " << MinBOSCCCost
                    << ", min. all-zero mask prop.: "
                    << MinBOSCCJumpOverProbability << "\n");

  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>> RPOT(
      Plan.getVectorLoopRegion());
  for (auto [Mask, Entry, Exiting] : SESEs) {
    auto *Pred = dyn_cast_or_null<VPBasicBlock>(Entry->getSinglePredecessor()),
         *Succ = dyn_cast_or_null<VPBasicBlock>(Exiting->getSingleSuccessor());
    // TODO: Relax to Succ->getNumPredecessors() > 1
    if (!Pred || Pred->getNumSuccessors() != 1 || !Succ ||
        Succ->getNumPredecessors() != 1 ||
        vputils::isUniformAcrossVFsAndUFs(Mask))
      continue;
    LLVM_DEBUG(dbgs() << "BOSCC: Candiate: %" << Entry->getName() << " -> %"
                      << Exiting->getName() << "\n");
    assert(DT.dominates(Pred, Succ) && DT.dominates(Entry, Exiting));

    // Get a estimate of the probability to execute these instructions.
    // TODO: Bypass cost checks in case of `hasBranchWeightsOrigin(Br)`?
    auto EntryProp = CostCtx.getEntryProbability(Entry->getEntryBasicBlock());
    if (EntryProp.isUnknown()) {
      LLVM_DEBUG(dbgs() << "BOSCC: Unknwon entry prop.\n");
      continue;
    }
    auto NotProp = BranchProbability::getOne() - EntryProp;
    auto MaskZeroProp = BranchProbability::getOne();
    for (unsigned I = 0; I < VFxIC; ++I)
      MaskZeroProp *= NotProp;
    if (MaskZeroProp < MinBOSCCJumpOverProbability) {
      LLVM_DEBUG(dbgs() << "BOSCC: All-zero mask prop. too low: "
                        << MaskZeroProp << "\n");
      continue;
    }

    // Calcuate the cost of this subregion.
    InstructionCost Cost = 0;
    auto EntryPos = find(RPOT, Entry),
         ExitingPos = std::find(EntryPos, RPOT.end(), Exiting);
    assert(EntryPos != RPOT.end() && ExitingPos != RPOT.end());
    auto SESERange = make_range(EntryPos, ExitingPos + 1);
    for (VPBlockBase *BB : SESERange)
      Cost += BB->cost(VF, CostCtx) * IC;
    if (!Cost.isValid() || Cost < MinBOSCCCost) {
      LLVM_DEBUG(dbgs() << "BOSCC: Cost too low: " << Cost << "\n");
      continue;
    }

    // Modify the VPlan and insert the BOSCC branch:
    LLVM_DEBUG(dbgs() << "BOSCC: Add BOSCC branch for %" << Entry->getName()
                      << " -> %" << Exiting->getName()
                      << " region: estimates: VFxIC is ~" << VFxIC << ", Cost: "
                      << Cost << "\n       entry prop.: " << EntryProp
                      << ", all-zero mask prop.: " << MaskZeroProp << "\n");
    VPBlockUtils::connectBlocks(Pred, Succ);
    DT.applyUpdates({{VPDominatorTree::Insert, Pred, Succ}});
    // The mask used for the entry block of the SESE can always be hoisted out,
    // and that same mask is often also used in the successor block.
    // TODO: At the moment, the VPlan-based cost-model ignores mask-creation
    // costs. When that changes, then this hoist should probably done before the
    // cost of the SESE is computed.
    hoistMasksInFrontOf(Mask, DT, *Pred, Pred->end());
    VPBuilder Builder(Pred, Pred->end());
    Builder.createNaryOp(VPInstruction::BranchOnCond,
                         {Builder.createNaryOp(VPInstruction::AnyOf, {Mask})});

    // Handle values leaving the jumped-over region.
    // PHIs will have to be created for them. Poison can be used as values for
    // the edge from the block with the BOSCC branch.
    for (VPBasicBlock *BB : VPBlockUtils::blocksOnly<VPBasicBlock>(SESERange))
      for (VPRecipeBase &Def : *BB)
        for (VPValue *V : Def.definedValues()) {
          SmallVector<VPRecipeBase *> UsesToFix;
          for (VPUser *U : V->users())
            if (auto *UR = dyn_cast<VPRecipeBase>(U);
                UR && !DT.dominates(BB, UR->getParent()) && !UR->isPhi())
              UsesToFix.push_back(UR);
          if (UsesToFix.empty())
            continue;

          PHINode *IRPhi = nullptr;
          if (auto *Blend = dyn_cast<VPBlendRecipe>(UsesToFix[0]);
              Blend && any_of(index_range(0, Blend->getNumIncomingValues()),
                              [&](unsigned I) {
                                return Blend->getIncomingValue(I) == V;
                              }))
            IRPhi = cast<PHINode>(Blend->getUnderlyingValue());

          Type *Ty = CostCtx.Types.inferScalarType(V);
          auto *Phi = new VPWidenPHIRecipe(IRPhi);
          Phi->addOperand(V);
          Phi->addOperand(Plan.getOrAddLiveIn(PoisonValue::get(Ty)));
          Phi->insertBefore(*Succ, Succ->getFirstNonPhi());
          for (VPRecipeBase *UR : UsesToFix)
            for (unsigned I = 0; I < UR->getNumOperands(); ++I)
              if (UR->getOperand(I) == V)
                UR->setOperand(I, Phi);
        }
  }

  assert(DT.verify(VPDominatorTree::VerificationLevel::Fast));
  assert(verifyVPlanIsValid(Plan));
}
