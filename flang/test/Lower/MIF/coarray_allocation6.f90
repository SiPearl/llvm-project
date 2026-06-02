! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program alloc_test
  type :: my_type2
    integer, allocatable :: co[:]
  end type
  
  type :: my_type
    type(my_type2) :: z
  end type
  
  type :: my_type3
    integer, allocatable :: w(:)
  end type
  
  type :: my_type4
    integer, pointer :: ptr => null()
  end type

  integer :: s
  character(len=10) :: msg
  type(my_type) :: a
  type(my_type3) :: b[*]
  type(my_type4) :: c[*]

  
  ! CHECK: %[[VAL_1:.*]] = hlfir.designate %[[ADDR_1:.*]]{"co"} fortran_attrs<allocatable> : (!fir.ref<!fir.type<_QFTmy_type2{co:!fir.box<!fir.heap<i32>, corank:1>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
  ! CHECK:  mif.alloc_coarray %[[VAL_1]] lcobounds %[[LCOBOUNDS:.*]] ucobounds %[[UCOBOUNDS:.*]] errmsg %[[ERRMSG:.*]] uniq_name("_QFEa.z.co") : (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>, !fir.box<none>) -> ()
  allocate(a%z%co[*])

  ! CHECK: %[[VAL_3:.*]] = hlfir.designate %[[ADDR_2:.*]]{"w"}   fortran_attrs<allocatable> : (!fir.heap<!fir.type<_QFTmy_type3{w:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: mif.alloc %[[VAL_3]] errmsg %[[ERRMSG:.*]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<none>) -> ()
  allocate(b%w(100))

  ! CHECK: %[[VAL_4:.*]] = hlfir.designate %[[ADDR_3:.*]]{"ptr"}   fortran_attrs<pointer> : (!fir.heap<!fir.type<_QFTmy_type4{ptr:!fir.box<!fir.ptr<i32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  ! CHECK: mif.alloc %[[VAL_4]] stat %[[STAT:.*]] errmsg %[[ERRMSG:.*]] : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<i32>, !fir.box<!fir.char<1,10>>) -> ()
  allocate(c%ptr, STAT=s, errmsg=msg)

  ! CHECK: mif.dealloc_coarray %[[ADDR_4:.*]] stat %[[STAT:.*]] errmsg %[[ERRMSG:.*]] : (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>, !fir.ref<i32>, !fir.box<none>) -> ()
  deallocate(a%z%co)
  
  ! CHECK: %[[VAL_5:.*]] = hlfir.designate %[[ADDR_5:.*]]{"w"}   fortran_attrs<allocatable> : (!fir.heap<!fir.type<_QFTmy_type3{w:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: mif.dealloc %[[VAL_5]] stat %[[STAT:.*]] errmsg %[[ERRMSG:.*]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<i32>, !fir.box<!fir.char<1,10>>) -> ()
  deallocate(b%w, STAT=s, errmsg=msg)

  ! CHECK: %[[VAL_6:.*]] = hlfir.designate %[[ADDR_6:.*]]{"ptr"}   fortran_attrs<pointer> : (!fir.heap<!fir.type<_QFTmy_type4{ptr:!fir.box<!fir.ptr<i32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  ! CHECK: mif.dealloc %[[VAL_6]] stat %[[STAT:.*]] errmsg %[[ERRMSG:.*]] : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<i32>, !fir.box<none>) -> ()
  deallocate(c%ptr)
end program

