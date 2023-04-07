#!/bin/bash

set -e

cd $(dirname $0)

nosetests-2.7 -v
