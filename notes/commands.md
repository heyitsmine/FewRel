# 常用命令

## 训练

## 测试

```shell
python train_demo.py --test semeval_233 --model proto-plain --encoder cnn --trainN 20 --batch_size 4 --only_test --load_ckpt ./checkpoint/proto-plain-cnn-train_wiki-val_wiki-5-5.pth.tar
```

## 生成测试数据

```shell
python sample_io.py ../data/val_wiki.json 5000 5 5 233 input > wiki_233_input.json
```

## 监控

```shell
echo $[$(cat /sys/class/thermal/thermal_zone0/temp)/1000]°
nvidia-smi -q -i 0,1 -d TEMPERATURE
```

