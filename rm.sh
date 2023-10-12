#!/bin/bash

TARGET=$1

docker run --rm -v $(pwd):/app -w /app alpine chmod -R 777 $TARGET
rm -fri $TARGET
