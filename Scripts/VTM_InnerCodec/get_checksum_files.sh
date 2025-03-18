#!/bin/bash

for FILE in $(find . -type f \( -name '*.bin' -o -name '*.yuv' \) ); do
  DIR=$(dirname "$FILE")
  FILENAME=$(basename "$FILE")
  #echo $DIR $FILENAME
  pushd "$DIR"
  md5sum -b "$FILENAME" > "$FILENAME.md5"
  popd
done



