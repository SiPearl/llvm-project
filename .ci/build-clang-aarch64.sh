#!/usr/bin/env bash

# Define sysroot before project.sh to set SW_VERSION
sysroot=${SYSROOT:-/workspace/sipearl/tools/sysroots/debian-11.0.0-arm64}
. $(dirname "$0")/project.sh

set -eux

# Create directories
artifacts_dir=$(createDir ${artifacts_dir})
install_prefix=$(createDir ${install_prefix})
install_dir=$(createDir ${install_dir})

build_directory=${BUILD_DIR:-"build_llvm"}
build_directory=$(createDir ${current_directory}/${build_directory})

TARGETS_TO_BUILD=${TARGETS_TO_BUILD:-'all'}
TARGET_TRIPLE=${TARGET_TRIPLE:-'aarch64-linux-gnu'}

echo "Build LLVM in ${build_directory}"
echo "Install path: ${install_prefix}"

# module dependencies
module_load gcc

CC=$(which aarch64-linux-gnu-gcc)
CXX=$(which aarch64-linux-gnu-g++)

GBU_PREFIX=$(dirname $(which aarch64-linux-gnu-ld))
if [ -z "${GBU_PREFIX}" ]; then
    echo "ERROR: GBU linker aarch64-linux-gnu-ld not found"
    exit -1
fi
BINUTILS_INCDIR=${GBU_PREFIX}/../x86_64-pc-linux-gnu/aarch64-linux-gnu/include

if [ ! -f ${BINUTILS_INCDIR}/plugin-api.h ]; then
    echo "ERROR: Unable to find plugin-api.h header in ${BINUTILS_INCDIR} path"
    echo "   -> GNU binutils must be built using: --enable-gold --enable-plugins configure options"
    exit -1
fi

pushd ${build_directory}
  ${CMAKE} -S ../llvm -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${install_dir} -DLLVM_INSTALL_UTILS=On -DCMAKE_CXX_STANDARD=17 \
        -DLLVM_ENABLE_ASSERTIONS=On -DLLVM_ENABLE_DUMP=On -DLLVM_BUILD_TESTS=On \
        -DCMAKE_C_COMPILER="${CC}" -DCMAKE_C_COMPILER_LAUNCHER="${CCACHE}" \
        -DCMAKE_CXX_COMPILER="${CXX}" -DCMAKE_CXX_COMPILER_LAUNCHER="${CCACHE}" \
        -DLLVM_ENABLE_PROJECTS="${LLVM_PROJECTS}" \
        -DLLVM_TARGETS_TO_BUILD="${TARGETS_TO_BUILD}" \
        -DDEFAULT_SYSROOT="${sysroot}" \
        -DLLVM_BINUTILS_INCDIR=${BINUTILS_INCDIR} \
        -DBUILD_SHARED_LIBS=ON \
        -DLLVM_DEFAULT_TARGET_TRIPLE="${TARGET_TRIPLE}"

  cp ./CMakeCache.txt ${artifacts_dir}/llvm-CMakeCache.txt
  make -j ${jobs}
  make install install-clang
popd

# Install of lit
python3 -m pip install --user ./llvm/utils/lit

echo "Test LLVM in ${build_directory}"
LIT_TEST_DIRS=${LIT_TEST_DIRS:-"test test/Unit"}

for test_dir in ${LIT_TEST_DIRS} ; do
    report_name="lit-report.${test_dir//\//_}.xml"
    ${build_directory}/bin/llvm-lit -j ${jobs} -s --xunit-xml-output ${artifacts_dir}/${report_name} ${build_directory}/${test_dir}
done

generate_modulefile "${install_prefix}/${package_prefix}/modulefiles/${SW_NAME,,}/${SW_VERSION}" \
    "${SW_NAME}" "${SW_LONG_NAME}" "${SW_VERSION}" "${SW_CATEGORY}" \
    "${SW_DESCRIPTION}" "${SW_INSTALL_SUFFIX}" "gcc"

echo "Size of artifacts: $(du -h -d 0 ./artifacts)"
