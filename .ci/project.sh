#!/usr/bin/env bash

set -eux

. $(dirname "$0")/checkout_aci.sh

. ${ACI}/module.sh
. ${ACI}/version.sh
. ${ACI}/fileutils.sh
. ${ACI}/deploy.sh
. ${ACI}/checkout.sh

SW_NAME=LLVM
SW_LONG_NAME="LLVM Compiler Infrastructure"
SW_CATEGORY="compilers"
SW_DESCRIPTION="LLVM, providing clang, clang++ and flang-new from LLVM Compiler Infrastructure"

sysroot=${sysroot:-""}
SW_VERSION=$(get_version "llvmorg-" "${sysroot}")
SW_INSTALL_SUFFIX=${SW_INSTALL_SUFFIX:-"${SW_NAME,,}/${SW_VERSION}"}
echo "Version: ${SW_VERSION}"

LLVM_PROJECTS=${LLVM_PROJECTS:-"clang;mlir;flang;openmp;clang-tools-extra"}

CMAKE=${CMAKE:-/toolsroot/lnx/cad/cmake-3.21.4-linux-x86_64/bin/cmake}
CCACHE=${CCACHE:-/home_spl/etienne.renault/shared/opt/x86/ccache-4.8.2-install/bin/ccache}

current_directory=$(pwd)
artifacts=${ARTIFACTS:-"./artifacts"}
package_prefix="opt/sipearl"
install_prefix=${INSTALL_PREFIX:-"install_llvm"}
jobs=${JOBS:-20}

artifacts_dir=${current_directory}/${artifacts}
install_prefix=${artifacts_dir}/${install_prefix}
install_dir=${install_prefix}/${package_prefix}/${SW_INSTALL_SUFFIX}
