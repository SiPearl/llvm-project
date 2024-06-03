#!/usr/bin/env bash

set -eux

from_sha1=${1:?"Must provide from sha1 (ref sha1)"}
to_sha1=${2:?"Must provide to sha1 (last sha1)"}
git_clang_format=${3:-"/workspace/sipearl/tools/x86_64/centos7.9-64/internal/opt/sipearl/clang/18.0.0/bin/git-clang-format"}

clang_path=$(dirname ${git_clang_format})
clang_path=$(realpath ${clang_path})

if [ -z "${clang_path}" -o ! -d ${clang_path} ]; then
    echo "Must provide valid git-clang-format: path ${clang_path} does not exist."
    exit -1
fi

if [ ! -f "${clang_path}/git-clang-format" ]; then
    echo "Must provide valid git-clang-format: ${clang_path}/git-clang-format does not exist."
    exit -1
fi

export PATH=${clang_path}:${PATH}
${clang_path}/git-clang-format --diff ${from_sha1} ${to_sha1} > ./clang-format.diff
