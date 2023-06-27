arg=10.0
act_prob=1.0
decay=0.998
nc=20
local_lr=0.1
pm=10.0
ST=42000
for algo in MGE
do
  for data in CIFAR10
  do
    time=$(date "+%Y-%m-%d %H:%M:%S")
    timer_start=`date "+%Y-%m-%d %H:%M:%S"`
    CUDA_VISIBLE_DEVICES=0 nohup python train.py --method $algo --dataset $data --rule-arg $arg --n_client $nc \
    --act_prob $act_prob --test-per 1 --lr-decay $decay --local-learning-rate $local_lr \
    --pm $pm --S_T $ST --epochs 1000 > $data-$arg-$algo-$local_lr-$time.log 2>&1
    timer_end=`date "+%Y-%m-%d %H:%M:%S"`
    duration=`echo $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`
    echo "开始： $timer_start"
    echo "结束： $timer_end"
    echo "$algo " "$data" " 耗时： $duration"
  done
done