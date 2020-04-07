#!/bin/bash
eval $(cd && .tspkg/bin/tsp --env)

echo "to trian model......" 
./scripts/infra_horovodrun.sh

echo "to test model........."
./scripts/infra_test.sh