// RUN: fir-opt --mif-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 22.0.0 (git@github.com:SiPearl/llvm-project.git 666e4313ebc03587f27774139ad8f780bac15c3e)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QQmain() attributes {fir.bindc_name = "TEST"} {
    %0 = fir.alloca !fir.array<1xi64>
    %1 = fir.alloca !fir.array<2xi64>
    %2 = fir.dummy_scope : !fir.dscope
    %3 = fir.address_of(@_QFEa) : !fir.ref<i32>
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %4 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64 to %4 : !fir.ref<i64>
    %c1_i64_0 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %5 = fir.coordinate_of %1, %c1 : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64_0 to %5 : !fir.ref<i64>
    %6 = fir.embox %1 : (!fir.ref<!fir.array<2xi64>>) -> !fir.box<!fir.array<2xi64>>
    %c1_i64_1 = arith.constant 1 : i64
    %c0_2 = arith.constant 0 : index
    %7 = fir.coordinate_of %0, %c0_2 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64_1 to %7 : !fir.ref<i64>
    %8 = fir.embox %0 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
    mif.alloc_coarray %3 lcobounds %6 ucobounds %8 {uniq_name = "_QFEa"} : (!fir.ref<i32>, !fir.box<!fir.array<2xi64>>, !fir.box<!fir.array<1xi64>>) -> ()
    %9:2 = hlfir.declare %3 {fir.corank = 2 : i32, uniq_name = "_QFEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %10 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
    %11:2 = hlfir.declare %10 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %c2 = arith.constant 2 : index
    %12 = fir.alloca !fir.array<2xi32> {bindc_name = "j", uniq_name = "_QFEj"}
    %13 = fir.shape %c2 : (index) -> !fir.shape<1>
    %14:2 = hlfir.declare %12(%13) {uniq_name = "_QFEj"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
    %15 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}> {bindc_name = "team", uniq_name = "_QFEteam"}
    %16:2 = hlfir.declare %15 {uniq_name = "_QFEteam"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>)
    %17 = fir.address_of(@_QQ_QM__fortran_builtinsT__builtin_team_type.DerivedInit) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    fir.copy %17 to %16#0 no_overlap : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %18 = mif.this_image : () -> i32
    hlfir.assign %18 to %11#0 : i32, !fir.ref<i32>
    %19 = mif.this_image team %16#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> i32
    hlfir.assign %19 to %11#0 : i32, !fir.ref<i32>
    %20 = mif.this_image coarray %9#0 : (!fir.ref<i32>) -> !fir.box<!fir.array<?xi64>>
    %21:2 = hlfir.declare %20 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.array<?xi64>>) -> (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>)
    %false = arith.constant false
    %22 = hlfir.as_expr %21#0 move %false : (!fir.box<!fir.array<?xi64>>, i1) -> !hlfir.expr<?xi64>
    %c0_3 = arith.constant 0 : index
    %23:3 = fir.box_dims %21#0, %c0_3 : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
    %24 = fir.shape %23#1 : (index) -> !fir.shape<1>
    %25 = hlfir.elemental %24 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
    ^bb0(%arg0: index):
      %27 = hlfir.apply %22, %arg0 : (!hlfir.expr<?xi64>, index) -> i64
      %28 = fir.convert %27 : (i64) -> i32
      hlfir.yield_element %28 : i32
    }
    hlfir.assign %25 to %14#0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<2xi32>>
    hlfir.destroy %25 : !hlfir.expr<?xi32>
    hlfir.destroy %22 : !hlfir.expr<?xi64>
    %c1_i32 = arith.constant 1 : i32
    %26 = mif.this_image coarray %9#0 dim %c1_i32 : (!fir.ref<i32>, i32) -> i32
    hlfir.assign %26 to %14#0 : i32, !fir.ref<!fir.array<2xi32>>
    return
  }
}

// CHECK-LABEL: func.func @_QQmain
// CHECK: fir.call @_QMprifPprif_this_image_no_coarray
// CHECK: fir.call @_QMprifPprif_this_image_with_coarray
// CHECK: fir.call @_QMprifPprif_this_image_with_dim
