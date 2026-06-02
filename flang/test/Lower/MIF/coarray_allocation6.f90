! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program alloc_test
  type :: my_type2
    integer, allocatable :: co[:]
  end type
  
  type :: my_type
    integer :: x
    integer, allocatable :: y(:)
    type(my_type2) :: z
  end type
  
  type :: my_type3
    integer, allocatable :: w(:)
  end type
  
  type :: my_type4
    integer, pointer :: ptr => null()
  end type

  integer :: me
  type(my_type) :: a
  type(my_type3) :: b[*]
  type(my_type4) :: c[*]

  
  ! CHECK: %[[VAL_1:.*]] = hlfir.designate %[[VAL_0:.*]]{"co"} {fortran_attrs = #fir.var_attrs<allocatable>} :
  ! (!fir.ref<!fir.type<_QFTmy_type2{co:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
  ! CHECK:  mif.alloc_coarray %[[VAL_1]] lcobounds %[[LCOBOUNDS:.*]] ucobounds %[[UCOBOUNDS:.*]] errmsg %[[ERRMSG:.*]] {uniq_name = "_QFEa.z.co"} : (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>, !fir.box<none>) -> ()
  allocate(a%z%co[*])

  ! CHECK: %[[VAL_3:.*]] = hlfir.designate %[[ADDR_4:.*]]#0{"w"} {fortran_attrs = #fir.var_attrs<allocatable>} : ({{.*}}) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: mif.alloc %[[VAL_3]] errmsg %[[ERRMSG:.*]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<none>) -> ()
  allocate(b%w(100))

  c%ptr = me
end program
