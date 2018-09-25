sudo perf record -a -g ./naive_rotate
sudo perf script  -f -i perf.data -c naive_rotate &> perf.unfold
stackcollapse-perf.pl perf.unfold &> perf.folded
flamegraph.pl perf.folded > perf.svg
rm perf.data* perf.unfold perf.folded