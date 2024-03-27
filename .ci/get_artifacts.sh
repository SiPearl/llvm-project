#!/usr/bin/env bash

set -eu

. $(dirname "$0")/checkout_aci.sh

. ${ACI}/gitlab.sh

if [ -n "$1" ]; then
    JOB_ID=$1
fi

JOB_ID=${JOB_ID:?"You have to define JOB_ID of the artifacts you want to download"}

if [ -z "${JOB_ID}" ]; then
    echo "ERROR: JOB_ID empty. You have to define JOB_ID of the artifacts you want to download"
    exit -1
fi

get_artifacts "735" "9YJqG3zrsHvaKz9iYtAi" "llvm-project" "${JOB_ID}"
