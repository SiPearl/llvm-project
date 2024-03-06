//===--------------------- SchedulingInfoView.cpp --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the SchedulingInfoView API.
///
//===----------------------------------------------------------------------===//

#include "Views/SchedulingInfoView.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/JSON.h"

namespace llvm {
namespace mca {

StringRef SchedulingInfoView::getComment(const llvm::MCInst &MCI) const {
  std::string s = MCI.getLoc().getPointer();
  std::string InstrStr;
  size_t pos = 0, pos_cmt = 0;

  // Recognized comments are after assembly instructions on the same line.
  // It is usefull to add in comment scheduling information from architecture
  // specification.
  // '#' comment mark is not supported by llvm-mca

  CommentString = "";
  if ((pos = s.find("\n")) != std::string::npos) {
    InstrStr = s.substr(0, pos);
    // C style comment
    if (((pos_cmt = InstrStr.find("/*")) != std::string::npos) &&
        ((pos = InstrStr.find("*/")) != std::string::npos)) {
      CommentString = InstrStr.substr(pos_cmt, pos);
      return StringRef(CommentString);
    }
    // C++ style comment
    if ((pos_cmt = InstrStr.find("//")) != std::string::npos) {
      CommentString = InstrStr.substr(pos_cmt, pos);
      return StringRef(CommentString);
    }
  }
  return StringRef(CommentString);
}

void SchedulingInfoView::printView(raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  formatted_raw_ostream FOS(TempStream);

  ArrayRef<llvm::MCInst> Source = getSource();
  if (!Source.size())
    return;

  IIVDVec IIVD(Source.size());
  collectData(IIVD);

  FOS << "\n\nResources:\n";
  const MCSchedModel &SM = getSubTargetInfo().getSchedModel();
  for (unsigned I = 1, ResourceIndex = 0, E = SM.getNumProcResourceKinds();
       I < E; ++I) {
    const MCProcResourceDesc &ProcResource = *SM.getProcResource(I);
    unsigned NumUnits = ProcResource.NumUnits;
    // Skip invalid resources with zero units.
    if (!NumUnits)
      continue;

    FOS << '[' << ResourceIndex << ']';
    FOS.PadToColumn(6);
    FOS << "- " << ProcResource.Name << ':' << NumUnits;
    if (ProcResource.SubUnitsIdxBegin) {
      FOS.PadToColumn(20);
      for (unsigned U = 0; U < NumUnits; ++U) {
        FOS << SM.getProcResource(ProcResource.SubUnitsIdxBegin[U])->Name
            << ", ";
      }
    }
    FOS << '\n';
    ResourceIndex++;
  }

  FOS << "\n\nScheduling Info:\n";
  FOS << "[1]: #uOps\n[2]: Latency\n[3]: Bypass Latency\n"
      << "[4]: Throughput\n[5]: HasSideEffects (SE)\n"
      << "[6]: LLVM OpcodeName\n[7]: Resources\n"
      << "[8]: Instruction\n[9]: Comment if any\n";

  // paddings for each scheduling info output. Start at [2]
  std::vector<unsigned> paddings = {7, 12, 18, 27, 32, 63, 113, 150};
  for (unsigned i = 0; i < paddings.size(); i++) {
    FOS << "[" << i + 1 << "]";
    FOS.PadToColumn(paddings[i]);
  }
  FOS << "[" << paddings.size() + 1 << "]\n";

  for (const auto &[Index, IIVDEntry, Inst] : enumerate(IIVD, Source)) {
    FOS << ' ' << IIVDEntry.NumMicroOpcodes;
    FOS.PadToColumn(paddings[0]);
    FOS << ' ' << IIVDEntry.Latency;
    FOS.PadToColumn(paddings[1]);
    FOS << ' ' << IIVDEntry.Bypass;
    FOS.PadToColumn(paddings[2]);
    if (IIVDEntry.Throughput) {
      double RT = *IIVDEntry.Throughput;
      FOS << format(" %.2f", RT);
    } else {
      FOS << " - ";
    }
    FOS.PadToColumn(paddings[3]);
    FOS << (IIVDEntry.hasUnmodeledSideEffects ? " SE " : " - ");
    FOS.PadToColumn(paddings[4]);
    FOS << format(" %s", IIVDEntry.OpcodeName.c_str());
    FOS.PadToColumn(paddings[5]);
    FOS << format(" %s", IIVDEntry.Resources.c_str());
    FOS.PadToColumn(paddings[6]);
    FOS << ' ' << printInstructionString(Inst);
    FOS.PadToColumn(paddings[7]);
    FOS << ' ' << getComment(Inst) << '\n';
  }

  FOS.flush();
  OS << Buffer;
}

void SchedulingInfoView::collectData(
    MutableArrayRef<SchedulingInfoViewData> IIVD) const {
  const llvm::MCSubtargetInfo &STI = getSubTargetInfo();
  const MCSchedModel &SM = STI.getSchedModel();
  for (const auto &I : zip(getSource(), IIVD)) {
    const MCInst &Inst = std::get<0>(I);
    SchedulingInfoViewData &IIVDEntry = std::get<1>(I);
    const MCInstrDesc &MCDesc = MCII.get(Inst.getOpcode());

    // Obtain the scheduling class information from the instruction
    // and instruments.
    auto IVecIt = InstToInstruments.find(&Inst);
    unsigned SchedClassID =
        IVecIt == InstToInstruments.end()
            ? MCDesc.getSchedClass()
            : IM.getSchedClassID(MCII, Inst, IVecIt->second);
    unsigned CPUID = SM.getProcessorID();

    // Try to solve variant scheduling classes.
    while (SchedClassID && SM.getSchedClassDesc(SchedClassID)->isVariant())
      SchedClassID =
          STI.resolveVariantSchedClass(SchedClassID, &Inst, &MCII, CPUID);

    const MCSchedClassDesc &SCDesc = *SM.getSchedClassDesc(SchedClassID);
    IIVDEntry.OpcodeName = (std::string)MCII.getName(Inst.getOpcode());
    IIVDEntry.NumMicroOpcodes = SCDesc.NumMicroOps;
    IIVDEntry.Latency = MCSchedModel::computeInstrLatency(STI, SCDesc);
    IIVDEntry.Bypass =
        IIVDEntry.Latency - MCSchedModel::getBypassDelayCycles(STI, SCDesc);
    IIVDEntry.Throughput =
        1.0 / MCSchedModel::getReciprocalThroughput(STI, SCDesc);
    IIVDEntry.hasUnmodeledSideEffects = MCDesc.hasUnmodeledSideEffects();
    raw_string_ostream TempStream(IIVDEntry.Resources);

    const MCWriteProcResEntry *Index = STI.getWriteProcResBegin(&SCDesc);
    const MCWriteProcResEntry *Last = STI.getWriteProcResEnd(&SCDesc);
    for (; Index != Last; ++Index) {
      if (!Index->ReleaseAtCycle)
        continue;
      const MCProcResourceDesc *MCProc =
          SM.getProcResource(Index->ProcResourceIdx);
      if (Index->ReleaseAtCycle != 1) {
        // Output ReleaseAtCycle between [] if not 1 (default)
        TempStream << format("%s[%d]", MCProc->Name, Index->ReleaseAtCycle)
                   << ", ";
      } else {
        TempStream << format("%s", MCProc->Name) << ", ";
      }
    }
    TempStream.flush();
  }
}

// Construct a JSON object from a single SchedulingInfoViewData object.
json::Object
SchedulingInfoView::toJSON(const SchedulingInfoViewData &IIVD) const {
  json::Object JO({{"NumMicroOpcodes", IIVD.NumMicroOpcodes},
                   {"Latency", IIVD.Latency},
                   {"LatencyWithBypass", IIVD.Bypass},
                   {"hasUnmodeledSideEffects", IIVD.hasUnmodeledSideEffects}});
  JO.try_emplace("Throughput", IIVD.Throughput.value_or(0.0));
  return JO;
}

json::Value SchedulingInfoView::toJSON() const {
  ArrayRef<llvm::MCInst> Source = getSource();
  if (!Source.size())
    return json::Value(0);

  IIVDVec IIVD(Source.size());
  collectData(IIVD);

  json::Array InstInfo;
  for (const auto &I : enumerate(IIVD)) {
    const SchedulingInfoViewData &IIVDEntry = I.value();
    json::Object JO = toJSON(IIVDEntry);
    JO.try_emplace("Instruction", (unsigned)I.index());
    InstInfo.push_back(std::move(JO));
  }
  return json::Object({{"InstructionList", json::Value(std::move(InstInfo))}});
}
} // namespace mca.
} // namespace llvm
