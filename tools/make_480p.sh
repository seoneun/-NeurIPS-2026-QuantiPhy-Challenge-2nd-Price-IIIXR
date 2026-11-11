#!/bin/bash
R="$(cd "${QP_ROOT:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd)"; export QP_ROOT="$R"
[ -f "$R/tools/env.sh" ] && source "$R/tools/env.sh"
SRC=${SRC:-$R/data/test}; DST=${DST:-$R/data/test_480p}; MAN=$DST/_manifest.tsv
mkdir -p $DST
[ -s $MAN ] || printf "video\tsrc_w\tsrc_h\trotate\taction\tdst_w\tdst_h\tdst_packets\tsrc_fps\tdst_fps\tsec\n" > $MAN
VF="scale=w='if(lt(iw,ih),480,-2)':h='if(lt(iw,ih),-2,480)':flags=area+accurate_rnd+full_chroma_int"
echo "START $(date) $(ls $SRC/*.mp4 | wc -l) videos"; uptime
n=0; nf=0
for f in $SRC/*.mp4; do
  b=$(basename "$f")
  if [ -s "$DST/$b" ] && grep -q "^$b	" $MAN; then continue; fi
  t0=$(date +%s.%N)
  IFS=x read W H < <(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "$f")
  ROT=$(ffprobe -v error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1:nk=1 "$f"); ROT=${ROT:-0}
  SFPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=nw=1:nk=1 "$f")
  S=$(( W < H ? W : H ))
  if [ "$S" -le 480 ]; then
    cp -p "$f" "$DST/$b.part" && mv "$DST/$b.part" "$DST/$b"; rc=$?; act=copy
  else
    ffmpeg -nostdin -hide_banner -loglevel error -threads 2 -filter_threads 2 -noautorotate -i "$f" \
      -map 0:v:0 -an -sn -dn -vf "$VF" -vsync passthrough \
      -c:v libx264 -preset medium -crf 14 -pix_fmt yuv420p -threads 2 -movflags +faststart -f mp4 "$DST/$b.part" \
      && mv "$DST/$b.part" "$DST/$b"; rc=$?; act=x264
  fi
  if [ $rc -ne 0 ]; then echo "FAIL $b rc=$rc"; rm -f "$DST/$b.part"; nf=$((nf+1)); continue; fi
  IFS=x read DW DH DP < <(ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=width,height,nb_read_packets -of csv=p=0:s=x "$DST/$b")
  DFPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=nw=1:nk=1 "$DST/$b")
  dt=$(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b-a}')
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.1f\n" "$b" "$W" "$H" "$ROT" "$act" "$DW" "$DH" "$DP" "$SFPS" "$DFPS" "$dt" >> $MAN
  n=$((n+1)); [ $((n % 25)) -eq 0 ] && echo "$(date +%T) done $n (last $b ${W}x${H} -> ${DW}x${DH}, ${dt}s) | load:$(uptime | sed 's/.*load average://')"
done
echo "END $(date) new=$n failed=$nf total_in_manifest=$(($(wc -l < $MAN) - 1))"
du -sh --apparent-size $SRC $DST
