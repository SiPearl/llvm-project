! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program test
  use iso_fortran_env
  integer :: n, image_num
  integer, parameter :: const_integer = 1
  type(team_type) :: team

  ! CHECK: fir.call @_QMprifPprif_image_status(
  n = image_status(1)

  ! CHECK: fir.call @_QMprifPprif_image_status(
  n = image_status(const_integer)

  ! CHECK: fir.call @_QMprifPprif_image_status(
  n = image_status(image_num)

  ! CHECK: fir.call @_QMprifPprif_image_status(
  n = image_status(IMAGE=1,         TEAM=team)

  ! CHECK: fir.call @_QMprifPprif_image_status(
  n = image_status(IMAGE=image_num, TEAM=team)

end program
