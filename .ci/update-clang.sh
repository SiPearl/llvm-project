#!/usr/bin/env bash

. $(dirname "$0")/project.sh

set -eux

project_name=${SW_NAME}

sha1=$(git rev-parse HEAD)
prj_branch="$(dirname "$0")/projects_branches.txt"

prj_ref="$(dirname "$0")/projects_ref_updates.txt"

gen_mod_files=${gen_mod_file_pattern/\*/}
rm -f ${gen_mod_files}
for gen_file in $(ls ${gen_mod_file_pattern}); do
    echoinfo "Get generated module file: ${gen_file}"
    cat ${gen_file} >> ${gen_mod_files}
done

updating_revfiles_in_projects "${project_name}" "${prj_branch}" "" "${prj_ref}" "${gen_mod_files}" "${sha1}"

