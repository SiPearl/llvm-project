#!/usr/bin/env bash

set -eux

. $(dirname "$0")/project.sh

echo "Deploying LLVM"
echo "Install path: ${install_dir}"

deploy "${install_prefix}/${package_prefix}" "${SW_INSTALL_SUFFIX}"
