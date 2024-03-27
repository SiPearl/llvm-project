#!/usr/bin/env bash

# Define sysroot before project.sh to set SW_VERSION
sysroot=${SYSROOT:-native}
. $(dirname "$0")/project.sh

set -eux

# Create directories
artifacts_dir=$(createDir ${artifacts_dir})
install_prefix=$(createDir ${install_prefix})
install_dir=$(createDir ${install_dir})

TARGETS_TO_BUILD=${TARGETS_TO_BUILD:-'all'}
TARGET_TRIPLE=${TARGET_TRIPLE:-'x86_64-pc-linux-gnu'}

build_directory=${BUILD_DIR:-"build_llvm_${TARGET_TRIPLE}"}
build_directory=$(createDir ${current_directory}/${build_directory})

echo "Build LLVM in ${build_directory}"
echo "Install path: ${install_prefix}"

# module dependencies
module_load gcc-x86
module_load gbu-${TARGET_TRIPLE}

CC=$(which gcc)
CXX=$(which g++)

sysroot_option=""
linker=ld
if [ ! -z "${sysroot}" -a "${sysroot}" != "native" ]; then
    if [ ! -d "${sysroot}" ]; then
	echo "ERROR: sysroot directory: ${sysroot} does not exist"
	exit -1
    fi
    sysroot_option="-DDEFAULT_SYSROOT=${sysroot}"
    linker_name=${TARGET_TRIPLE}-ld
fi

GBU_PREFIX=$(dirname $(which ${linker}))
if [ -z "${GBU_PREFIX}" ]; then
    echo "ERROR: GBU linker ${linker} not found"
    exit -1
fi
BINUTILS_INCDIR=${GBU_PREFIX}/../include

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
        -DLLVM_BINUTILS_INCDIR=${BINUTILS_INCDIR} \
        -DBUILD_SHARED_LIBS=ON ${sysroot_option} \
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
    "${SW_DESCRIPTION}" "${SW_INSTALL_SUFFIX}" "gcc-x86 gbu-${TARGET_TRIPLE}"

echo "Size of artifacts: $(du -h -d 0 ./artifacts)"
