#!/bin/bash
# Delete /dev/shm/psm_* (multiprocessing.shared_memory) segments that no live process maps or holds open and that are
# older than 10 min. DeepInfer servers run with --ipc=host, so every killed server leaks its 8 GiB segments (#5).
#   bash scripts/clean_orphan_shm.sh            # dry run
#   bash scripts/clean_orphan_shm.sh --delete
used=$(sudo bash -c 'for p in /proc/[0-9]*; do cat $p/maps 2>/dev/null | grep -o "/dev/shm/psm_[^ ]*"; ls -l $p/fd 2>/dev/null | grep -o "/dev/shm/psm_[^ ]*"; done' | sort -u)
n=0; b=0; k=0
for f in $(find /dev/shm -maxdepth 1 -name "psm_*" -mmin +10); do
  if grep -qxF "$f" <<< "$used"; then k=$((k+1)); continue; fi
  s=$(stat -c %s "$f"); n=$((n+1)); b=$((b+s))
  [ "$1" = --delete ] && sudo rm -f "$f"
done
echo "in use: $k   orphan: $n ($((b/1000000000)) GB) $( [ "$1" = --delete ] && echo deleted)"
