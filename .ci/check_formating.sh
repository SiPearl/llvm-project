#!/usr/bin/env bash

set -eux

from_sha1=${1:?"Must provide from sha1 (ref sha1)"}
to_sha1=${2:?"Must provide to sha1 (last sha1)"}
git_clang_format=${3:-"/workspace/sipearl/tools/x86_64/centos7.9-64/internal/opt/sipearl/clang/18.0.0/bin/git-clang-format"}

sha1_exists() {
    local sha1=${1:?"Unknown sha1"}

    if (git cat-file -e ${sha1} 2>1 > /dev/null) then
	echo "true"
    fi
    echo "false"
}

if [ "${from_sha1}" -eq 0 ]; then
    from_sha1="${to_sha1}^"
fi

to_sha1_exists=$(sha1_exists ${to_sha1})
if [ "${to_sha1_exists}" = "false" ]; then
	echo "ERROR: ${to_sha1} does not exist"
	exit -1
fi
# From sha1 not exist due to push force and replacement
# of this sha1 with new one (to_sha1)
from_sha1_exists=$(sha1_exists ${from_sha1})
if [ "${from_sha1_exists}" = "false" ]; then
    from_sha1="${to_sha1}^"
fi

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

first_sipearl_commit=""
last_sipearl_commit=""
for sha1 in $(git log --pretty=format:"%H;;%ae" ${from_sha1}..${to_sha1}); do
    fields=(${sha1/;;/ })
    first_sipearl_commit=${fields[0]}
    if [[ ${sha1} =~ sipearl\.com$ ]]; then
	if [ -z "${last_sipearl_commit}" ]; then
	    last_sipearl_commit=${fields[0]}
	fi
    else
	break
    fi
done

if [ -z "${first_sipearl_commit}" -o -z "${last_sipearl_commit}" ]; then
    echo "ERROR: Unable to find a list of sipearl commits"
    exit -1
fi

if [ "${first_sipearl_commit}" = "${last_sipearl_commit}" ]; then
    first_sipearl_commit="${first_sipearl_commit}^"
fi


export PATH=${clang_path}:${PATH}
${clang_path}/git-clang-format --diff ${first_sipearl_commit} ${last_sipearl_commit} > ./clang-format.diff
