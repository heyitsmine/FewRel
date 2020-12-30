import json
import sys
import random

# 构造size个N-way, K-shot任务，每个任务的查询集有一个样本
# python sample_io.py ../data/val_wiki.json 5000 5 5 233 input > wiki_233_input.json
if len(sys.argv) != 7:
    print("Usage: python sample_io.py filename.json size N K seed input/output")
filename = sys.argv[1]
size = int(sys.argv[2])
N = int(sys.argv[3])
K = int(sys.argv[4])
seed = int(sys.argv[5])
io = sys.argv[6]
random.seed(seed)

whole_division = json.load(open(filename))
relations = whole_division.keys()

input_data = []
output_data = []
for i in range(size):
    sampled_relation = random.sample(relations, N) # 从所有类别中选出K个
    target = random.choice(range(len(sampled_relation))) # 从[0, N) 中随机选取1个作为查询集类别
    output_data.append(target)
    target_relation = sampled_relation[target]
    meta_train = [random.sample(whole_division[i], K) for i in sampled_relation] # list[list[sample]], 选取N*K个样本
    meta_test = random.choice(whole_division[target_relation]) # 随机选取1个查询集样本
    input_data.append({"meta_train": meta_train, "meta_test": meta_test})

# input: list[size * {'meta_train': list[N * list[K * sample]], 'meta_test': sample}]
# output: list[size * int]
if io == "input":
    json.dump(input_data, sys.stdout)
else:
    json.dump(output_data, sys.stdout)
