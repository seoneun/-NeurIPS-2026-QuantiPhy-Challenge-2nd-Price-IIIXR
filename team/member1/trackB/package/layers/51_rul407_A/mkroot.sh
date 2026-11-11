#!/bin/bash
set -e; SI=$(cd "$1" && pwd); FR=$2; QR=$3; TOP=$(pwd); mkdir -p "$QR"; QR=$(cd "$QR" && pwd)
cp -as "$SI/." "$QR/"
while read -r f; do [ -n "$f" ] || continue; mkdir -p "$(dirname "$QR/$f")"; ln -s "$TOP/$f" "$QR/$f"; done < "$FR"
