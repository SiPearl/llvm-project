#!/usr/bin/env bash

set -eux

. $(dirname "$0")/checkout_aci.sh

. ${ACI}/sipearl_aci.env

. $(dirname "$0")/options.sh

# Currently supported ACI API version to check here
export CURRENT_ACI_VERSION="0.1.1"

# Verify if check_version is available in sipearl_aci.env
# Only check_version will remain here when projects support version > 0.1.1
check_version ${ACI_VERSION} ${CURRENT_ACI_VERSION}

SW_NAME=LLVM
SW_LONG_NAME="LLVM Compiler Infrastructure"
SW_CATEGORY="compilers"
SW_DESCRIPTION="LLVM, providing clang, clang++ and flang-new from LLVM Compiler Infrastructure"

sysroot=${SYSROOT:-native}
SW_VERSION=$(get_version "llvmorg-" "${sysroot}")
SW_INSTALL_SUFFIX=${SW_INSTALL_SUFFIX:-"${SW_NAME,,}/${SW_VERSION}"}
echo "Version: ${SW_VERSION}"

CMAKE=${CMAKE:-/toolsroot/lnx/cad/cmake-3.21.4-linux-x86_64/bin/cmake}
CCACHE=${CCACHE:-/home_spl/etienne.renault/shared/opt/x86/ccache-4.8.2-install/bin/ccache}

current_directory=$(pwd)
artifacts=${ARTIFACTS:-"./artifacts"}
install_prefix=${INSTALL_PREFIX:-"install_llvm"}
jobs=${JOBS:-40}

artifacts_dir=${current_directory}/${artifacts}
install_prefix=${artifacts_dir}/${install_prefix}
install_dir=${install_prefix}/${SW_INSTALL_SUFFIX}
