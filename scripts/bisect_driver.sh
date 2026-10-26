#!/bin/bash
W=/opt/dlami/nvme/bench_tmp; SRC=$W/Mooncake-bisect
[ -d $SRC ] || git clone -q https://github.com/kvcache-ai/Mooncake.git $SRC
cd $SRC && git bisect reset -q 2>/dev/null; git bisect start 83f8f877 dd1217b && git bisect run bash $W/bisect.sh > $W/logs/bisect_run.log 2>&1
git bisect log >> $W/logs/bisect_run.log; echo BISECT_DONE >> $W/logs/bisect_run.log
