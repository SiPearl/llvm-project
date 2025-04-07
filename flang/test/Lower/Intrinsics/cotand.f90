! RUN: bbc -emit-hlfir %s -o - | FileCheck %s --check-prefixes="CHECK,CHECK-FAST"
! RUN: bbc --math-runtime=precise -emit-hlfir %s -o - | FileCheck %s --check-prefixes="CHECK,CHECK-PRECISE"
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefixes="CHECK,CHECK-FAST"

subroutine test_real4(x)
  real :: x, res
  res = cotand(x)
end subroutine

! CHECK-LABEL: @_QPtest_real4
! CHECK: %[[dfactor:.*]] = arith.constant 0.017453292519943295 : f64
! CHECK: %[[factor:.*]] = fir.convert %[[dfactor]] : (f64) -> f32
! CHECK: %[[arg:.*]] = arith.mulf %{{[A-Za-z0-9._]+}}, %[[factor]] fastmath<contract> : f32
! CHECK-PRECISE: %[[tand:.*]] = fir.call @tanf(%[[arg]]) fastmath<contract> : (f32) -> f32
! CHECK-FAST: %[[tand:.*]] = math.tan %[[arg]] fastmath<contract> : f32
! CHECK: %[[const:.*]] = arith.constant 1 : i32
! CHECK: %[[const_f:.*]] = fir.convert %[[const]] : (i32) -> f32
! CHECK: %[[res:.*]] = arith.divf %[[const_f]], %[[tand]] fastmath<contract> : f32

subroutine test_real8(x)
  real(8) :: x, res
  res = cotand(x)
end subroutine

! CHECK-LABEL: @_QPtest_real8
! CHECK: %[[factor:.*]] = arith.constant 0.017453292519943295 : f64
! CHECK: %[[arg:.*]] = arith.mulf %{{[A-Za-z0-9._]+}}, %[[factor]] fastmath<contract> : f64
! CHECK-PRECISE: %[[tand:.*]] = fir.call @tan(%[[arg]]) fastmath<contract> : (f64) -> f64
! CHECK-FAST: %[[tand:.*]] = math.tan %[[arg]] fastmath<contract> : f64
! CHECK: %[[const:.*]] = arith.constant 1 : i32
! CHECK: %[[const_f:.*]] = fir.convert %[[const]] : (i32) -> f64
! CHECK: %[[res:.*]] = arith.divf %[[const_f]], %[[tand]] fastmath<contract> : f64

