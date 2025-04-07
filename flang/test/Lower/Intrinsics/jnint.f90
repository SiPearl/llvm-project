! RUN: bbc -emit-hlfir %s -o - | FileCheck %s


! CHECK-LABEL: func @_QPtest_jnint_real
! CHECK-SAME: %[[ARG_0:.*]]: !fir.ref<f64>{{.*}}, %[[ARG_1:.*]]: !fir.ref<i64>
subroutine test_jnint_real(a,r)
! CHECK: %[[DEC_0:.*]]:2 = hlfir.declare %[[ARG_0]]
! CHECK: %[[DEC_1:.*]]:2 = hlfir.declare %[[ARG_1]]
! CHECK: %[[VAL_0:.*]] = fir.load %[[DEC_0]]#0 : !fir.ref<f64>
! CHECK: %[[VAL_1:.*]] = fir.call @llvm.lround.i32.f64(%[[VAL_0]])
! CHECK: %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (i32) -> i64
! CHECK: hlfir.assign %[[VAL_2]] to %[[DEC_1]]#0
! CHECK: return
              real(kind=8) :: a
              integer(kind=8) :: r
              r = JNINT(a)
end subroutine
