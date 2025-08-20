! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=ALL,COARRAY
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefixes=ALL,NOCOARRAY

! ALL-LABEL: func @_QPfail_image_test
subroutine fail_image_test

  fail image
! COARRAY-NOT: fir.call @_FortranAFailImageStatement()
! COARRAY: fir.call @_QMprifPprif_fail_image() {{.*}}:

! NOCOARRAY-NOT: fir.call @_QMprifPprif_fail_image
! NOCOARRAY: fir.call @_FortranAFailImageStatement() {{.*}}:
! ALL:  fir.unreachable
end subroutine 
