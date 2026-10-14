; RUN: opt -disable-output -passes=loop-vectorize -vectorizer-div-vf-by=2 -force-vector-interleave=2 -scalable-vectorization=preferred %s -pass-remarks=VFDivisorWarning 2>&1 | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

; Warnings are emitted in function order.
; CHECK: warning:{{.*}}-vectorizer-div-vf-by was not applied: find-first/last-index reductions are not supported
; CHECK: warning:{{.*}}-vectorizer-div-vf-by was not applied: unsupported partial reduction
; CHECK: warning:{{.*}}-vectorizer-div-vf-by: interleave count forced to 1, first-order recurrences are not supported with interleaving

define i64 @find_iv(ptr noalias %a, i32 %x, i64 %n) #0 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %rdx = phi i64 [ -1, %entry ], [ %sel, %loop ]
  %gep = getelementptr inbounds i32, ptr %a, i64 %iv
  %v = load i32, ptr %gep, align 4
  %cmp = icmp eq i32 %v, %x
  %sel = select i1 %cmp, i64 %iv, i64 %rdx
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret i64 %sel
}

define i64 @partial_red(ptr noalias %a, i64 %n) #0 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %sum = phi i64 [ 0, %entry ], [ %add, %loop ]
  %gep = getelementptr inbounds i32, ptr %a, i64 %iv
  %v = load i32, ptr %gep, align 4
  %ext = sext i32 %v to i64
  %add = add i64 %sum, %ext
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret i64 %add
}

define void @recurrence(i64 %n, ptr noalias %a, ptr %c) #0 {
entry:
  %c.first = load i64, ptr %c, align 8
  br label %loop

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %rec = phi i64 [ %c.load, %loop ], [ %c.first, %entry ]
  %res = add i64 %rec, 42
  %c.addr = getelementptr inbounds i64, ptr %c, i64 %i
  %c.load = load i64, ptr %c.addr, align 8
  %a.addr = getelementptr inbounds i64, ptr %a, i64 %i
  store i64 %res, ptr %a.addr, align 8
  %i.next = add nuw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

attributes #0 = { vscale_range(1,16) "target-features"="+sve2" }
