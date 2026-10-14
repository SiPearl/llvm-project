; RUN: not opt -S -passes=loop-vectorize -vectorizer-div-vf-by=3 %s 2>&1 | FileCheck %s
; RUN: not opt -S -passes=loop-vectorize -vectorizer-div-vf-by=6 %s 2>&1 | FileCheck %s --check-prefix=SIX

; CHECK: -vectorizer-div-vf-by must be 0 or a power of 2, got 3
; CHECK-NOT: Stack dump
; CHECK-NOT: PLEASE submit a bug report
; SIX: got 6

define void @f() {
  ret void
}
