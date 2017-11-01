#!/bin/bash

NPROCS=1
BINARY="numactl -m 1 python caffe2_mnist.py"

nprocs=$(grep '^physical id' /proc/cpuinfo  | sort -u | wc -l)
ncores=$(grep '^processor' /proc/cpuinfo | sort -u | wc -l)
coresperproc=$((ncores/nprocs))
OMP_NUM_THREADS=$((NPROCS*coresperproc))

freesock=$(./getfreesocket -explicit=${NPROCS})
if [ "z$freesock" == "z" ]
then
  echo "Not enough free processors!  aborting"
  exit 1
else
  KMP_AFFINITY="granularity=fine,proclist=[$freesock],explicit"
  GOMP_CPU_AFFINITY="$(echo $freesock | sed -e 's/,/ /g')"
fi

export KMP_AFFINITY OMP_NUM_THREADS GOMP_CPU_AFFINITY
echo "KMP_AFFINITY:"${KMP_AFFINITY} 
echo "OMP_NUM_THREADS:"${OMP_NUM_THREADS} 
echo "GOMP_CPU_AFFINITY:"${GOMP_CPU_AFFINITY}

#${BINARY}
