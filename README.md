# HashRetrieval
Very Fast and Low-Memory Open-Domain Retrieval Dialog Systems by Using Semantic Hashing

现有的检索式开放域对话系统都主要分为两个阶段，分别是粗筛检索和精细排序。粗筛阶段目前都是使用BM25为主的term frequency的检索方法从预构建的语料库中选取和上下文主题比较相似的作为候选回复，精细排序阶段使用语义匹配模型从所有的候选回复中选取最合适的一个。目前大家的研究方向都放在构建一个更加精准的精细排序模型，但是目前，随着以bert为主的预训练模型逐渐在各个数据集的各个指标上取得了目前最好的成绩，精细排序模型的性能提升变得非常的困难，这使得提升检索式对话系统的主要瓶颈局限于如何构建一个更好更快的粗筛检索模块上。

但是目前的开放域对话系统的粗筛检索模块中，大家基本上都是用的是基于term frequency的方法，这种方法会召回和上下文具有相同的词或者词组的回复，在QA等其他任务中这样的粗筛检索模块是有效的，这是因为具有和问题一样的词的答案极大概率就是包含有正确答案的那个。但是这在开放域对话系统中却并不一定，在开放域对话系统中，和上下文具有相同的词或者词组（主题）的句子未必就是最合适的恢复，这使得使用传统的粗筛检索模块并不能有效的选出最合适的候选回复，从而导致性能的下降，如下所示（仍然需要数据支撑这个观点）：

* 上下文：我最喜欢的看惊悚和恐怖电影了
* 回复：我喜欢看恐怖电影
* 潜在候选回复：你为什么喜欢看这种猎奇类型呢

目前，real-vector检索已被证明可以提升QA系统的效果，但是real-vector是否可以有效的提升开放域对话系统仍然是一个待研究的问题。其次，目前的real-vector检索面临的主要问题就是存储空间大和查询速度相对慢的问题。我们也要提出一个基于Hash的vector语义检索模型，期望可以在不损失大量的效果的前提下，可以获得极快的查询速度和极低的存储大小，以促进检索式对话系统的实际应用和部署，比如移动设备上等。

## 1. How to Use
### 1.1 Train the dual-bert model

dual-bert has two bert model, the batch size is 16

```bash
# for example: ./run.sh train ecommerce dual-bert 0,1,2,3
./run.sh train <dataset_name> dual-bert <gpu_ids>
```

### 1.2 Train the cross-bert model

cross-bert has one bert mode, the batch size is 32

```bash
./run.sh train <dataset_name> cross-bert <gpu_ids>
```

### 1.3 Generate the vector for the corpus

dual-bert and hash-bert generate the vectors of the utterances in the corpus

```bash
./run.sh inference <dataset_name> dual-bert/hash-bert <gpu_ids>
```

## 2. Experiment Results
### 2.1 Comparsion between Term-Frequency and Dense vector retrieval
Compare the performance and the time cost (The pre-constructed corpus is the combination of the train and test dataset)
* Elasticsearch v.s. Faiss-cpu (IndexFlatL2)

| Method | Top-20 | Top-100 | Time Cost (batch=32) |
| :----: | :----: | :-----: | :------------------: |
| Dense  | 0.18   | 0.345   |   1.0119s            |
| BM25   | 0.022  | 0.04    |   0.1789s            |

### 2.2 Comparsion between the Dense vector and Hash vector retrieval
Compare the performance and the time cost

### 2.3 Overall comparsion
cross-bert post rank with different coarse retrieval strategies:
* Term-Frequency
* Dense vector
* Hash vector