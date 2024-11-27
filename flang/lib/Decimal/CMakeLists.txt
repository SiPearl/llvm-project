#===-- lib/Decimal/CMakeLists.txt ------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

set(FORTRAN_DECIMAL_TYPE CACHE STRING "")
add_flang_library(FortranDecimal ${FORTRAN_DECIMAL_TYPE}
  binary-to-decimal.cpp
  decimal-to-binary.cpp
)
