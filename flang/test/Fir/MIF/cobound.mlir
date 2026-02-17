// RUN: fir-opt --mif-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 23.0.0 (git@github.com:SiPearl/llvm-project.git d31a4730513391710d91c5ad33bb8ea3d68db3cb)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
// CHECK-LABEL: func.func @_QQmain
  func.func @_QQmain() attributes {fir.bindc_name = "TEST"} {
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@_QFEa) : !fir.ref<!fir.box<!fir.heap<i32>>>
    %2:2 = hlfir.declare %1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
    %c3 = arith.constant 3 : index
    %3 = fir.alloca !fir.array<3xi32> {bindc_name = "res1", uniq_name = "_QFEres1"}
    %4 = fir.shape %c3 : (index) -> !fir.shape<1>
    %5:2 = hlfir.declare %3(%4) {uniq_name = "_QFEres1"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
    %6 = fir.alloca i32 {bindc_name = "res2", uniq_name = "_QFEres2"}
    %7:2 = hlfir.declare %6 {uniq_name = "_QFEres2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %8 = fir.absent !fir.box<none>
    mif.alloc_coarray %2#0 errmsg %8 {lcobounds = array<i64: 1, 3, 1>, ucobounds = array<i64: 2, 5, -1>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<none>) -> ()
    %9 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
    %10 = fir.box_addr %9 {fir.corank = 3 : i32} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
// CHECK: fir.call @_QMprifPprif_lcobound_no_dim
    %11 = mif.lcobound coarray %10 : (!fir.heap<i32>) -> !fir.box<!fir.array<?xi64>>
    %12:2 = hlfir.declare %11 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.array<?xi64>>) -> (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>)
    %false = arith.constant false
    %13 = hlfir.as_expr %12#0 move %false : (!fir.box<!fir.array<?xi64>>, i1) -> !hlfir.expr<?xi64>
    %c0 = arith.constant 0 : index
    %14:3 = fir.box_dims %12#0, %c0 : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
    %15 = fir.shape %14#1 : (index) -> !fir.shape<1>
    %16 = hlfir.elemental %15 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
    ^bb0(%arg0: index):
      %31 = hlfir.apply %13, %arg0 : (!hlfir.expr<?xi64>, index) -> i64
      %32 = fir.convert %31 : (i64) -> i32
      hlfir.yield_element %32 : i32
    }
    hlfir.assign %16 to %5#0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<3xi32>>
    hlfir.destroy %16 : !hlfir.expr<?xi32>
    hlfir.destroy %13 : !hlfir.expr<?xi64>
    %c2_i32 = arith.constant 2 : i32
    %17 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
    %18 = fir.box_addr %17 {fir.corank = 3 : i32} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
// CHECK: fir.call @_QMprifPprif_lcobound_with_dim
    %19 = mif.lcobound coarray %18 dim %c2_i32 : (!fir.heap<i32>, i32) -> i32
    hlfir.assign %19 to %7#0 : i32, !fir.ref<i32>
    %20 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
    %21 = fir.box_addr %20 {fir.corank = 3 : i32} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
// CHECK: fir.call @_QMprifPprif_ucobound_no_dim
    %22 = mif.ucobound coarray %21 : (!fir.heap<i32>) -> !fir.box<!fir.array<?xi64>>
    %23:2 = hlfir.declare %22 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.array<?xi64>>) -> (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>)
    %false_0 = arith.constant false
    %24 = hlfir.as_expr %23#0 move %false_0 : (!fir.box<!fir.array<?xi64>>, i1) -> !hlfir.expr<?xi64>
    %c0_1 = arith.constant 0 : index
    %25:3 = fir.box_dims %23#0, %c0_1 : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
    %26 = fir.shape %25#1 : (index) -> !fir.shape<1>
    %27 = hlfir.elemental %26 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
    ^bb0(%arg0: index):
      %31 = hlfir.apply %24, %arg0 : (!hlfir.expr<?xi64>, index) -> i64
      %32 = fir.convert %31 : (i64) -> i32
      hlfir.yield_element %32 : i32
    }
    hlfir.assign %27 to %5#0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<3xi32>>
    hlfir.destroy %27 : !hlfir.expr<?xi32>
    hlfir.destroy %24 : !hlfir.expr<?xi64>
    %c2_i32_2 = arith.constant 2 : i32
    %28 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
    %29 = fir.box_addr %28 {fir.corank = 3 : i32} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
// CHECK: fir.call @_QMprifPprif_ucobound_with_dim
    %30 = mif.ucobound coarray %29 dim %c2_i32_2 : (!fir.heap<i32>, i32) -> i32
    hlfir.assign %30 to %7#0 : i32, !fir.ref<i32>
    return
  }
}
