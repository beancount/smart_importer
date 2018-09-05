#!/usr/bin/env bash

VENV="/Users/johannes/repos/beancount/virtualenv"

pushd `dirname $0` > /dev/null
source ${VENV}/bin/activate
bean-extract main.import.py ./downloads -f ledger/example.beancount
popd > /dev/null
