#!/bin/bash
set -e
PKG=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"; O=$(cd "$1" && pwd)
for st in 2_c2_g3a 3_g3a_m3 4_m3_z1s2c; do bash "$PKG/$st/run.sh" "$O"; done
