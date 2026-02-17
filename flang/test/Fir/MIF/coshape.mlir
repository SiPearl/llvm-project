// RUN: fir-opt --mif-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 23.0.0 (git@github.com:SiPearl/llvm-project.git d31a4730513391710d91c5ad33bb8ea3d68db3cb)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QQmain() attributes {fir.bindc_name = "TEST"} {
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@_QFEa) : !fir.ref<i32>
    mif.alloc_coarray %1 {lcobounds = array<i64: 1, 3, 1>, ucobounds = array<i64: 2, 5, -1>, uniq_name = "_QFEa"} : (!fir.ref<i32>) -> ()
    %2:2 = hlfir.declare %1 {fir.corank = 3 : i32, uniq_name = "_QFEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %c3 = arith.constant 3 : index
    %3 = fir.alloca !fir.array<3xi32> {bindc_name = "res", uniq_name = "_QFEres"}
    %4 = fir.shape %c3 : (index) -> !fir.shape<1>
    %5:2 = hlfir.declare %3(%4) {uniq_name = "_QFEres"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
    %c3_0 = arith.constant 3 : index
    %6 = fir.alloca !fir.array<3xi64> {bindc_name = "res2", uniq_name = "_QFEres2"}
    %7 = fir.shape %c3_0 : (index) -> !fir.shape<1>
    %8:2 = hlfir.declare %6(%7) {uniq_name = "_QFEres2"} : (!fir.ref<!fir.array<3xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi64>>, !fir.ref<!fir.array<3xi64>>)
    %9 = mif.coshape coarray %2#0 : (!fir.ref<i32>) -> !fir.box<!fir.array<?xi64>>
    %10:2 = hlfir.declare %9 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.array<?xi64>>) -> (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>)
    %false = arith.constant false
    %11 = hlfir.as_expr %10#0 move %false : (!fir.box<!fir.array<?xi64>>, i1) -> !hlfir.expr<?xi64>
    %c0 = arith.constant 0 : index
    %12:3 = fir.box_dims %10#0, %c0 : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
    %13 = fir.shape %12#1 : (index) -> !fir.shape<1>
    %14 = hlfir.elemental %13 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
    ^bb0(%arg0: index):
      %21 = hlfir.apply %11, %arg0 : (!hlfir.expr<?xi64>, index) -> i64
      %22 = fir.convert %21 : (i64) -> i32
      hlfir.yield_element %22 : i32
    }
    hlfir.assign %14 to %5#0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<3xi32>>
    hlfir.destroy %14 : !hlfir.expr<?xi32>
    hlfir.destroy %11 : !hlfir.expr<?xi64>
    %15 = mif.coshape coarray %2#0 : (!fir.ref<i32>) -> !fir.box<!fir.array<?xi64>>
    %16:2 = hlfir.declare %15 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.array<?xi64>>) -> (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>)
    %false_1 = arith.constant false
    %17 = hlfir.as_expr %16#0 move %false_1 : (!fir.box<!fir.array<?xi64>>, i1) -> !hlfir.expr<?xi64>
    %c0_2 = arith.constant 0 : index
    %18:3 = fir.box_dims %16#0, %c0_2 : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
    %19 = fir.shape %18#1 : (index) -> !fir.shape<1>
    %20 = hlfir.elemental %19 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi64> {
    ^bb0(%arg0: index):
      %21 = hlfir.apply %17, %arg0 : (!hlfir.expr<?xi64>, index) -> i64
      hlfir.yield_element %21 : i64
    }
    hlfir.assign %20 to %8#0 : !hlfir.expr<?xi64>, !fir.ref<!fir.array<3xi64>>
    hlfir.destroy %20 : !hlfir.expr<?xi64>
    hlfir.destroy %17 : !hlfir.expr<?xi64>
    return
  }
}

// CHECK-LABEL: func.func @_QQmain
// CHECK: fir.call @_QMprifPprif_coshape
// CHECK: fir.call @_QMprifPprif_coshape
