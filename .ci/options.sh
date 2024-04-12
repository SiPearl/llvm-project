#!/usr/bin/env bash

set -eux

host=$(uname -m)
if [ -n "${1:-""}" ]; then
    case ${1} in
	aarch64)
	    if [ "${host}" = "aarch64" ]; then
		export SYSROOT=native
		export TARGETS_TO_BUILD=all
	    else
		export SYSROOT=/workspace/sipearl/tools/sysroots/debian-11.0.0-arm64
		export TARGETS_TO_BUILD=AArch64
	    fi
	    export TARGET_TRIPLE=aarch64-linux-gnu
	    ;;
	x86_64)
	    if [ "${host}" = "x86_64" ]; then
		export SYSROOT=native
		export TARGETS_TO_BUILD=all
	    else
		echo "Error: Cross compilation host ${host} for target x86_64 not yet supported"
		exit -1
	    fi
	    export TARGET_TRIPLE=x86_64-pc-linux-gnu
	;;
	*)
	    echo "Unknown argument: '${1}'. Should be 'aarch64' or 'x86_64'"
	    exit 1
	    ;;
    esac
else
    # Default: build for native
    export SYSROOT=native
    export TARGETS_TO_BUILD=all
    case ${host} in
	aarch64)
	    export TARGET_TRIPLE=aarch64-linux-gnu
	    ;;
	x86_64)
	    export TARGET_TRIPLE=x86_64-pc-linux-gnu
	    ;;
	*)
	    echo "Not managed host: ${host}"
	    exit -1
	    ;;
    esac
fi
