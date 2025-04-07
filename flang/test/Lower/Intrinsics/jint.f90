! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtest_jint_int
! CHECK-SAME: %[[ARG_0:.*]]: !fir.ref<i32>{{.*}}, %[[ARG_1:.*]]: !fir.ref<i32>{{.*}}
subroutine test_jint_int(a,r)
! CHECK: %[[VAL_0:.*]] = fir.load %[[ARG_0]] : !fir.ref<i32>
! CHECK: fir.store %[[VAL_0]] to %[[ARG_1]] : !fir.ref<i32>
! CHECK: return
              integer:: a
              integer :: r
              r = JINT(a)
end subroutine

! CHECK-LABEL: func @_QPtest_jint_real
! CHECK-SAME: %[[ARG_0:.*]]: !fir.ref<f32>{{.*}}, %[[ARG_1:.*]]: !fir.ref<i32>
subroutine test_jint_real(a,r)
! CHECK: %[[VAL_0:.*]] = fir.load %[[ARG_0]] : !fir.ref<f32>
! CHECK: %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (f32) -> i32
! CHECK: fir.store %[[VAL_1]] to %[[ARG_1]] : !fir.ref<i32>
! CHECK: return
              real :: a
              integer :: r
              r = JINT(a)
end subroutine

! CHECK-LABEL: func @_QPtest_jint_complex
! CHECK-SAME:  %[[ARG_0:.*]]: !fir.ref<!fir.complex<4>>{{.*}}, %[[ARG_1:.*]]: !fir.ref<i32>{{.*}}) {
subroutine test_jint_complex(a,r)
! CHECK: %[[VAL_0:.*]] = fir.load %[[ARG_0]] : !fir.ref<!fir.complex<4>>
! CHECK: %[[VAL_1:.*]] = fir.extract_value %[[VAL_0]], [0 : index] : (!fir.complex<4>) -> f32
! CHECK: %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (f32) -> i32
! CHECK: fir.store %[[VAL_2]] to %[[ARG_1]] : !fir.ref<i32>
! CHECK: return
              complex:: a
              integer :: r
              r = JINT(a)
end subroutine

