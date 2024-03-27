#!/usr/bin/env bash

# Define sysroot before project.sh to set SW_VERSION
sysroot=${SYSROOT:-native}
. $(dirname "$0")/project.sh

set -eux

TARGET_TRIPLE=${TARGET_TRIPLE:-'x86_64-pc-linux-gnu'}

build_directory=${BUILD_DIR:-"llvm-test-suite-build-${TARGET_TRIPLE}"}
build_directory=$(createDir ${current_directory}/${build_directory})

CMAKE=${CMAKE:-/toolsroot/lnx/cad/cmake-3.21.4-linux-x86_64/bin/cmake}

echo "Test LLVM in ${build_directory}"

# module dependencies
module_load gcc-x86
module_load gbu-${TARGET_TRIPLE}

current_dir=$(pwd)

checkout llvm-test-suite https://gitlab01.int.sipearl.com/software/compilers/benchmarks/llvm-test-suite.git

# Test on X86 host for aarch64 or x86
# If host is equal to target, do not set OPTFLAGS. It will be defined to native by default
mcpu_flags=""
if [ "${TARGET_TRIPLE}" = "aarch64-linux-gnu" ]; then
    mcpu_flags="-DOPTFLAGS=-mcpu=neoverse-v1"
fi

${CMAKE} -S ${current_dir}/llvm-test-suite -B ${build_directory} -G 'Unix Makefiles' -DLLVM_INSTALL_DIR=${install_dir} ${mcpu_flags} -C ${current_dir}/llvm-test-suite/cmake/caches/SiPearl-CI.cmake
make -C ${build_directory} -j ${jobs} 
python3 -m pip install --user ./llvm/utils/lit
lit -j ${jobs} -s --xunit-xml-output ${artifacts_dir}/test-suite-report.xml ${build_directory}

