! Ensure argument -fallow-argument-mismatch works as expected.

! RUN: %flang -fallow-argument-mismatch -c %s 2>&1 | FileCheck %s --check-prefix=ALLOW
! RUN: not %flang %s -c 2>&1 | FileCheck %s --check-prefix=NOT-ALLOW

! ALLOW: warning: Reference to the procedure 'check_int' has an implicit interface that is distinct from another reference
! NOT-ALLOW: Semantic errors in 

integer function kind_of_int()
  integer(2), dimension(2) :: x2 = (/1, 2/)
  integer(4), dimension(2) :: x4 = (/1, 2/)
  character(len=1) :: ret

  kind_of_int=-1
  call check_int(x2(1),x2(2),ret)
  if (ret == 't') then
   kind_of_int=2
   return
  endif

  call check_int(x4(1),x4(2),ret)
  if (ret == 't') then
   kind_of_int=4
   return
  endif

end function kind_of_int

program same_int
  integer ki,kl

  ki=kind_of_int()
  if (ki /= kl) then
    write (*,'(i1)') 0
  else
    write (*,'(i1)') 1
  endif
end program same_int
