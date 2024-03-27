#!/usr/bin/env bash

set -eu

if [ -n "${CI_JOB_TOKEN:-""}" ]; then
    set -x
fi

script_dir=$(dirname "$0")
script_dir=$(realpath ${script_dir})

CI_JOB_TOKEN=${CI_JOB_TOKEN:-""}

if [ -f "${script_dir}/revfiles/aci.src" ]; then
    revision=$(cat ${script_dir}/revfiles/aci.src)

    echo "== Info: checkout ACI revision ${revision}"
    pushd ${script_dir}
      if [[ -z "${CI_JOB_TOKEN}" ]]; then
	  if [ -d aci ]; then
	      pushd aci
	        git fetch
	      popd
	  else
	      git clone https://gitlab01.int.sipearl.com/software/compilers/aci.git
	  fi
      else
	  rm -rf aci
	  git clone https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab01.int.sipearl.com/software/compilers/aci.git
      fi
      pushd aci
        git checkout ${revision}
      popd
    popd
else
    echo "${script_dir}/revfiles/aci.src does not exist. Unable to checkout aci repository"
    exit -1
fi

echo "== ACI is now revision ${revision}"
export ACI=${script_dir}/aci

