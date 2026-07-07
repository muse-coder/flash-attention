set +e
OUT=/tmp/ncu_heads_sweep.txt
echo "HQ HKV seqlen FA4_us FI_us FI/FA4" > $OUT
med() { python3 -c "import csv,sys,statistics;d=[float(r[-1]) for r in csv.reader(sys.stdin) if r and r[0].isdigit() and 'gpu__time' in r[-3]];print(round(statistics.median(d)/1000,1)) if d else print('NA')"; }
for HEADS in "16 1" "8 1" "16 2"; do set -- $HEADS; HQ=$1; HKV=$2
 for S in 8192 12288 16384; do
  FA=$(PYTHONPATH=/home/mudi/flash-attention CUTE_DSL_ARCH=sm_103a FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 CUDA_VISIBLE_DEVICES=0 B=1 S=$S HQ=$HQ HKV=$HKV timeout 400 ncu -k "regex:hd256" -c 3 --metrics gpu__time_duration.sum --csv /home/mudi/mudi_env/bin/python /tmp/fa4_one_param.py 2>/dev/null | med)
  FI=$(PYTHONPATH=/home/mudi/flashinfer CUDA_VISIBLE_DEVICES=0 B=1 S=$S HQ=$HQ HKV=$HKV timeout 400 ncu -k "regex:fmha" -c 3 --metrics gpu__time_duration.sum --csv /home/mudi/mudi_env/bin/python /tmp/fi_one_param.py 2>/dev/null | med)
  SP=$(python3 -c "print(f'{$FI/$FA:.2f}x') if '$FA'!='NA' and '$FI'!='NA' and $FA>0 else print('NA')" 2>/dev/null)
  echo "$HQ $HKV $S $FA $FI $SP" >> $OUT
 done
done
echo "=== DONE ===" >> $OUT
