#!/usr/bin/env bash

. $(dirname "$0")/project.sh

set -eux

# Create directories
artifacts_dir=$(createDir ${artifacts_dir})
install_prefix=$(createDir ${install_prefix})
install_dir=$(createDir ${install_dir})

build_directory=${BUILD_DIR:-"build_llvm_${TARGET_TRIPLE}"}
build_directory=$(createDir ${current_directory}/${build_directory})

echo "Build LLVM in ${build_directory}"
echo "Install path: ${install_prefix}"

sysroot=${SYSROOT:?}
sysroot_option=""
omp_option=""
libpfm=""
linker=ld
gbu_module=""
if [ -n "${sysroot}" -a "${sysroot}" != "native" ]; then
    if [ ! -d "${sysroot}" ]; then
	echo "ERROR: sysroot directory: ${sysroot} does not exist"
	exit -1
    fi

    sysroot_option="-DDEFAULT_SYSROOT=${sysroot}"
    omp_option=" -DLIBOMP_LDFLAGS=-Wl,-rpath-link=${sysroot}/lib/${TARGET_TRIPLE}"
    omp_option+=" -DLIBOMP_OMPD_SUPPORT=OFF"
    omp_option+=" -DLIBOMP_HAVE_SHM_OPEN_WITH_LRT=TRUE"
    linker_name=${TARGET_TRIPLE}-ld
    gbu_module="gbu-${TARGET_TRIPLE}"
else
    libpfm=" -DLLVM_ENABLE_LIBPFM=TRUE"
fi

# Generate meta module toolchain based on module dependencies
# Warning: This module list and make_toolroot is also in .ci/test-clang.sh
# TODO: Common make_toolroot. Artifact should be too big if we add toolroot in it...
make_toolroot "./toolroot" "papi-x86-native ${gbu_module} cmake-x86" ".ci/revfiles"
# Meta package is in ./toolroot/modulefiles and its name is toolchain/${SW_VERSION}
set +x
module use ./toolroot/modulefiles
module load toolchain/${SW_VERSION}
set -x

module_load python-x86

CC=$(which gcc)
CXX=$(which g++)

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

LLVM_PROJECTS=${LLVM_PROJECTS:-"clang;mlir;flang;clang-tools-extra"}
LLVM_RUNTIMES=${LLVM_RUNTIMES:-"openmp"}
BUILD_TYPE=${BUILD_TYPE:-"Release"}

pushd ${build_directory}
  cmake --trace-expand -S ../llvm -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_INSTALL_PREFIX=${package_prefix} -DLLVM_INSTALL_UTILS=On -DCMAKE_CXX_STANDARD=17 \
        -DLLVM_ENABLE_ASSERTIONS=On -DLLVM_ENABLE_DUMP=On -DLLVM_BUILD_TESTS=On \
        -DCMAKE_C_COMPILER="${CC}" \
        -DCMAKE_CXX_COMPILER="${CXX}" \
        -DLLVM_ENABLE_PROJECTS="${LLVM_PROJECTS}" \
        -DLLVM_ENABLE_RUNTIMES="${LLVM_RUNTIMES}" \
        -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
        -DLLVM_TARGETS_TO_BUILD="${TARGETS_TO_BUILD}" \
        -DLLVM_BINUTILS_INCDIR=${BINUTILS_INCDIR} \
        -DBUILD_SHARED_LIBS=ON ${sysroot_option} ${omp_option} ${libpfm} \
        -DPython3_EXECUTABLE=python3 \
        -DLLVM_DEFAULT_TARGET_TRIPLE="${TARGET_TRIPLE}" 2> ${artifacts_dir}/llvm-CMakeLogs.txt

  cp ./CMakeCache.txt ${artifacts_dir}/llvm-CMakeCache.txt
  make -j ${jobs}
  make DESTDIR=${install_dir} install install-clang
popd

if [ -n "${sysroot}" -a "${sysroot}" != "native" ]; then
    build_decimal=${BUILD_DIR:-"build_decimal_${TARGET_TRIPLE}"}
    build_decimal=$(createDir ${current_directory}/${build_decimal})

    pushd ${build_decimal}
      cmake -S ../flang/lib/Decimal -G "Unix Makefiles" \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DCMAKE_INSTALL_PREFIX=${package_prefix} -DCMAKE_CXX_STANDARD=17 \
          -DLLVM_ENABLE_ASSERTIONS=On \
	  -DCMAKE_C_COMPILER="${install_dir}${package_prefix}/bin/clang" \
	  -DCMAKE_CXX_COMPILER="${install_dir}${package_prefix}/bin/clang++" \
          -DLLVM_TARGETS_TO_BUILD="${TARGETS_TO_BUILD}" \
          -DBUILD_SHARED_LIBS=ON

      make -j ${jobs} FortranDecimal
      make DESTDIR=${install_dir} install
      cp ${build_decimal}/libFortranDecimal.a ${build_directory}/lib/
    popd

    build_runtime=${BUILD_DIR:-"build_runtime_${TARGET_TRIPLE}"}
    build_runtime=$(createDir ${current_directory}/${build_runtime})

    pushd ${build_runtime}
      cmake -S ../flang/runtime -G "Unix Makefiles" \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DCMAKE_INSTALL_PREFIX=${package_prefix} -DCMAKE_CXX_STANDARD=17 \
          -DLLVM_ENABLE_ASSERTIONS=On \
	  -DCMAKE_C_COMPILER="${install_dir}${package_prefix}/bin/clang" \
	  -DCMAKE_CXX_COMPILER="${install_dir}${package_prefix}/bin/clang++" \
          -DLLVM_TARGETS_TO_BUILD="${TARGETS_TO_BUILD}" \
          -DBUILD_SHARED_LIBS=ON

      make -j ${jobs} FortranRuntime
      make DESTDIR=${install_dir} install
      cp ${build_runtime}/libFortranRuntime.a ${build_directory}/lib/
    popd
fi

# Install of lit
python3 --version
python3 -m pip install --user ./llvm/utils/lit

echo "Test LLVM in ${build_directory}"
LIT_TEST_DIRS=${LIT_TEST_DIRS:-"test test/Unit tools/flang/test"}

for test_dir in ${LIT_TEST_DIRS} ; do
    report_name="lit-report.${test_dir//\//_}.xml"
    ${build_directory}/bin/llvm-lit -j ${jobs} -s --xunit-xml-output ${artifacts_dir}/${report_name} ${build_directory}/${test_dir}
done

plugin_file="${script_dir}/module_plugin.sh"
rm -f ${plugin_file}

echo "#!/usr/bin/env bash" > ${plugin_file}
echo "sw_plugin_prepend_path[LD_LIBRARY_PATH]='\$package_prefix/lib:\$package_prefix/lib/${TARGET_TRIPLE}'" > ${plugin_file}

generate_modulefile "${install_prefix}/modulefiles/${SW_NAME,,}/${SW_VERSION}" \
    "${SW_NAME}" "${SW_LONG_NAME}" "${SW_VERSION}" "${SW_CATEGORY}" \
    "${SW_DESCRIPTION}" "${SW_INSTALL_SUFFIX}" "papi-x86-native ${gbu_module}" "${plugin_file}" "${gen_mod_file}"

echo "Size of artifacts: $(du -h -d 0 ./artifacts)"
