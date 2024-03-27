#!/usr/bin/env bash

set -eux

# Define sysroot before project.sh to set SW_VERSION
sysroot=${SYSROOT:-""}
. $(dirname "$0")/project.sh

echo "Deploying LLVM"
echo "Install path: ${install_dir}"

deploy "${install_prefix}/${package_prefix}" "${SW_INSTALL_SUFFIX}"
