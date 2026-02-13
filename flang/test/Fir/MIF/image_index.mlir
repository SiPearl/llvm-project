// RUN: fir-opt --mif-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 23.0.0 (git@github.com:SiPearl/llvm-project.git d31a4730513391710d91c5ad33bb8ea3d68db3cb)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
// CHECK-LABEL: func.func @_QQmain
  func.func @_QQmain() attributes {fir.bindc_name = "TEST"} {
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@_QFEa) : !fir.ref<i32>
    mif.alloc_coarray %1 {lcobounds = array<i64: 1, 3, 1>, ucobounds = array<i64: 2, 5, -1>, uniq_name = "_QFEa"} : (!fir.ref<i32>) -> ()
    %2:2 = hlfir.declare %1 {uniq_name = "_QFEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3 = fir.address_of(@_QM__fortran_builtinsEC__builtin_atomic_int_kind) : !fir.ref<i32>
    %4:2 = hlfir.declare %3 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QM__fortran_builtinsEC__builtin_atomic_int_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %5 = fir.address_of(@_QM__fortran_builtinsEC__builtin_atomic_logical_kind) : !fir.ref<i32>
    %6:2 = hlfir.declare %5 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QM__fortran_builtinsEC__builtin_atomic_logical_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %20 = fir.alloca i32 {bindc_name = "idx", uniq_name = "_QFEidx"}
    %21:2 = hlfir.declare %20 {uniq_name = "_QFEidx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %157 = fir.address_of(@_QFEsub) : !fir.ref<!fir.array<3xi32>>
    %c3_1 = arith.constant 3 : index
    %158 = fir.shape %c3_1 : (index) -> !fir.shape<1>
    %159:2 = hlfir.declare %157(%158) {uniq_name = "_QFEsub"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
    %160 = fir.address_of(@_QFEsub2) : !fir.ref<!fir.array<3xi64>>
    %c3_2 = arith.constant 3 : index
    %161 = fir.shape %c3_2 : (index) -> !fir.shape<1>
    %162:2 = hlfir.declare %160(%161) {uniq_name = "_QFEsub2"} : (!fir.ref<!fir.array<3xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi64>>, !fir.ref<!fir.array<3xi64>>)
    %163 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}> {bindc_name = "team", uniq_name = "_QFEteam"}
    %164:2 = hlfir.declare %163 {uniq_name = "_QFEteam"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>)
    %165 = fir.address_of(@_QQ_QM__fortran_builtinsT__builtin_team_type.DerivedInit) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    fir.copy %165 to %164#0 no_overlap : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %178 = fir.shape %c3_1 : (index) -> !fir.shape<1>
    %179 = fir.embox %159#0(%178) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>

// CHECK: fir.call @_QMprifPprif_image_index
    %180 = mif.image_index coarray %2#0 sub %179 : (!fir.ref<i32>, !fir.box<!fir.array<3xi32>>) -> i32
    hlfir.assign %180 to %21#0 : i32, !fir.ref<i32>
    %181 = fir.shape %c3_2 : (index) -> !fir.shape<1>
    %182 = fir.embox %162#0(%181) : (!fir.ref<!fir.array<3xi64>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi64>>

// CHECK: fir.call @_QMprifPprif_image_index
    %183 = mif.image_index coarray %2#0 sub %182 : (!fir.ref<i32>, !fir.box<!fir.array<3xi64>>) -> i32
    hlfir.assign %183 to %21#0 : i32, !fir.ref<i32>
    %184 = fir.shape %c3_1 : (index) -> !fir.shape<1>
    %185 = fir.embox %159#0(%184) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>

// CHECK: fir.call @_QMprifPprif_image_index_with_team
    %186 = mif.image_index coarray %2#0 sub %185 team %164#0 : (!fir.ref<i32>, !fir.box<!fir.array<3xi32>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> i32
    hlfir.assign %186 to %21#0 : i32, !fir.ref<i32>
    return
  }
}
