// RUN: fir-opt --mif-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 22.0.0 (git@github.com:SiPearl/llvm-project.git 666e4313ebc03587f27774139ad8f780bac15c3e)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QQmain() attributes {fir.bindc_name = "TEST"} {
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@_QFEa) : !fir.ref<i32>
    mif.alloc_coarray %1 {lcobounds = array<i64: 1, 1>, ucobounds = array<i64: 2, -1>, uniq_name = "_QFEa"} : (!fir.ref<i32>) -> ()
    %2:2 = hlfir.declare %1 {uniq_name = "_QFEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
    %4:2 = hlfir.declare %3 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %c2 = arith.constant 2 : index
    %5 = fir.alloca !fir.array<2xi32> {bindc_name = "j", uniq_name = "_QFEj"}
    %6 = fir.shape %c2 : (index) -> !fir.shape<1>
    %7:2 = hlfir.declare %5(%6) {uniq_name = "_QFEj"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
    %8 = mif.this_image : () -> i32
    hlfir.assign %8 to %4#0 : i32, !fir.ref<i32>
    %9 = mif.this_image coarray %2#0 : (!fir.ref<i32>) -> !fir.box<!fir.array<?xi64>>
    %10:2 = hlfir.declare %9 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.array<?xi64>>) -> (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>)
    %false = arith.constant false
    %11 = hlfir.as_expr %10#0 move %false : (!fir.box<!fir.array<?xi64>>, i1) -> !hlfir.expr<?xi64>
    %c0 = arith.constant 0 : index
    %12:3 = fir.box_dims %10#0, %c0 : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
    %13 = fir.shape %12#1 : (index) -> !fir.shape<1>
    %14 = hlfir.elemental %13 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
    ^bb0(%arg0: index):
      %16 = hlfir.apply %11, %arg0 : (!hlfir.expr<?xi64>, index) -> i64
      %17 = fir.convert %16 : (i64) -> i32
      hlfir.yield_element %17 : i32
    }
    hlfir.assign %14 to %7#0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<2xi32>>
    hlfir.destroy %14 : !hlfir.expr<?xi32>
    hlfir.destroy %11 : !hlfir.expr<?xi64>
    %c1_i32 = arith.constant 1 : i32
    %15 = mif.this_image coarray %2#0 dim %c1_i32 : (!fir.ref<i32>, i32) -> i32
    hlfir.assign %15 to %7#0 : i32, !fir.ref<!fir.array<2xi32>>
    return
  }
}

// CHECK-LABEL: func.func @_QQmain
// CHECK: fir.call @_QMprifPprif_this_image_no_coarray
// CHECK: fir.call @_QMprifPprif_this_image_with_coarray
// CHECK: fir.call @_QMprifPprif_this_image_with_dim
