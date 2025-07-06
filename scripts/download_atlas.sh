#!/bin/bash

outputdir=$1
split=$2
# mkdir for output directory if it does not exist
mkdir -p ${outputdir}

# assert split in {atlas, atlas_train, atlas_val, atlas_test}
if [[ ${split} != "atlas" && ${split} != "atlas_train" && ${split} != "atlas_val" && ${split} != "atlas_test" ]]; then
    echo "Error: split must be one of {atlas, atlas_train, atlas_val, atlas_test}"
    exit 1
fi
workdir=$(pwd)

for name in $(cat `dirname $0`/../splits/${split}.csv | grep -v name | awk -F ',' {'print $1'}); do
    # wget https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/${name}/${name}_protein.zip -P ${outputdir}
    echo "Downloading ${name}... to ${outputdir}"
    cd ${outputdir}
    unzip -o ${name}_protein.zip
    rm ${name}_protein.zip
    cd ${workdir} # go back to the original directory
done