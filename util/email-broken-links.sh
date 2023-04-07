#!/usr/bin/bash

cd $(dirname $0)
git pull --rebase
OUT="$(./check-links.py)"

if [[ $? != 0 ]]; then
  echo "$OUT" | mutt brandon.amos.cs+openface.broken@gmail.com \
                     -s "Broken OpenFace Links"
fi

echo "$OUT"
