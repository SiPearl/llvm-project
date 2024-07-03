#!/usr/bin/env bash

set -eux

. $(dirname "$0")/project.sh

echo "Deploying LLVM"
echo "Install path: ${install_dir}"

deploy_from_generated_modules "${install_prefix}" "${gen_mod_file_pattern}"
