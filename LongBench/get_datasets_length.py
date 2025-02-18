from collections import defaultdict

import numpy as np
from datasets import load_dataset

datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec",
            "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
dataset_dict = defaultdict(list)
for dataset in datasets:
    data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
    for row in data:
        #if row['length'] < 4000:
        dataset_dict[dataset].append(row['length'])

for dataset in datasets:
    print(f"{dataset} mean: {np.mean(dataset_dict[dataset])}")
    print(f"{dataset} max: {np.max(dataset_dict[dataset])}")
    # print(f"{dataset} median: {np.median(dataset_dict[dataset])}")