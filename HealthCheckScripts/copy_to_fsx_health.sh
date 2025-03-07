#!/bin/bash

## quick set up healthcheck dependencies
GLOBAL_HEALTHCHECK_PATH=/fsx/healthcheck/
mkdir -p $GLOBAL_HEALTHCHECK_PATH
cp healthCheckMaster.sh $GLOBAL_HEALTHCHECK_PATH
cp healthCheckMaster.sh $GLOBAL_HEALTHCHECK_PATH

echo $GLOBAL_HEALTHCHECK_PATH
ls -l $GLOBAL_HEALTHCHECK_PATH