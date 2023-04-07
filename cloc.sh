#!/bin/bash

cloc batch-represent evaluation openface models training util \
  demos/*.py \
  demos/web/{*.{py,html,sh},js,css}
