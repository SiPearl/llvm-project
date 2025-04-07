! RUN: bbc -emit-hlfir %s -o - | FileCheck %s --check-prefixes="CHECK,CHECK-FAST"
! RUN: bbc --math-runtime=precise -emit-hlfir %s -o - | FileCheck %s --check-prefixes="CHECK,CHECK-PRECISE"
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefixes="CHECK,CHECK-FAST"

! CHECK-LABEL: @_QPtest_cotan_real4
! CHECK-SAME: %[[ARG_0:.*]]: !fir.ref<f32>{{.*}}, %[[ARG_1:.*]]: !fir.ref<f32>{{.*}}) {
subroutine test_cotan_real4(x,y)
! CHECK: %[[DEC_0:.*]]:2 = hlfir.declare %[[ARG_0]]
! CHECK: %[[DEC_1:.*]]:2 = hlfir.declare %[[ARG_1]]
! CHECK: %[[VAL_0:.*]] = fir.load %[[DEC_0]]#0 : !fir.ref<f32>
! CHECK-FAST: %[[VAL_1:.*]] = math.tan %[[VAL_0]] fastmath<contract> : f32
! CHECK-PRECISE: %[[VAL_1:.*]] = fir.call @tanf(%[[VAL_0]]) fastmath<contract> : (f32) -> f32
! CHECK: %[[CST_0:.*]] = arith.constant 1 : i32
! CHECK: %[[CST_1:.*]] = fir.convert %[[CST_0]] : (i32) -> f32
! CHECK: %[[VAL_2:.*]] = arith.divf %[[CST_1]], %[[VAL_1]] fastmath<contract> : f32
! CHECK: hlfir.assign %[[VAL_2]] to %[[DEC_1]]#0 : f32, !fir.ref<f32>
! CHECK: return
  real(4) :: x
  real(4) :: y
  y = cotan(x)
end subroutine test_cotan_real4

! CHECK-LABEL: @_QPtest_cotan_real
! CHECK-SAME: %[[ARG_0:.*]]: !fir.ref<f64>{{.*}}, %[[ARG_1:.*]]: !fir.ref<f64>{{.*}}) {
subroutine test_cotan_real(x,y)
! CHECK: %[[DEC_0:.*]]:2 = hlfir.declare %[[ARG_0]]
! CHECK: %[[DEC_1:.*]]:2 = hlfir.declare %[[ARG_1]]
! CHECK: %[[VAL_0:.*]] = fir.load %[[DEC_0]]#0 : !fir.ref<f64>
! CHECK-FAST: %[[VAL_1:.*]] = math.tan %[[VAL_0]] fastmath<contract> : f64
! CHECK-PRECISE: %[[VAL_1:.*]] = fir.call @tan(%[[VAL_0]]) fastmath<contract> : (f64) -> f64
! CHECK: %[[CST_0:.*]] = arith.constant 1 : i32
! CHECK: %[[CST_1:.*]] = fir.convert %[[CST_0]] : (i32) -> f64
! CHECK: %[[VAL_2:.*]] = arith.divf %[[CST_1]], %[[VAL_1]] fastmath<contract> : f64
! CHECK: hlfir.assign %[[VAL_2]] to %[[DEC_1]]#0 : f64, !fir.ref<f64>
! CHECK: return
  real(8) :: x
  real(8) :: y
  y = dcotan(x)
end subroutine test_cotan_real
