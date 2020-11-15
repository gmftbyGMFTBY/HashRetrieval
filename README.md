# HashRetrieval
Very Fast and Low-Memory Open-Domain Retrieval Dialog Systems by Using Semantic Hashing

现有的检索式开放域对话系统都主要分为两个阶段，分别是粗筛检索和精细排序。粗筛阶段目前都是使用BM25为主的term frequency的检索方法从预构建的语料库中选取和上下文主题比较相似的作为候选回复，精细排序阶段使用语义匹配模型从所有的候选回复中选取最合适的一个。目前大家的研究方向都放在构建一个更加精准的精细排序模型，但是目前，随着以bert为主的预训练模型逐渐在各个数据集的各个指标上取得了目前最好的成绩，精细排序模型的性能提升变得非常的困难，这使得提升检索式对话系统的主要瓶颈局限于如何构建一个更好更快的粗筛检索模块上。

但是目前的开放域对话系统的粗筛检索模块中，大家基本上都是用的是基于term frequency的方法，这种方法会召回和上下文具有相同的词或者词组的回复，在QA等其他任务中这样的粗筛检索模块是有效的，这是因为具有和问题一样的词的答案极大概率就是包含有正确答案的那个。但是这在开放域对话系统中却并不一定，在开放域对话系统中，和上下文具有相同的词或者词组（主题）的句子未必就是最合适的恢复，这使得使用传统的粗筛检索模块并不能有效的选出最合适的候选回复，从而导致性能的下降，如下所示（仍然需要数据支撑这个观点）：

* 上下文：我最喜欢的看惊悚和恐怖电影了
* 回复：我喜欢看恐怖电影
* 潜在候选回复：你为什么喜欢看这种猎奇类型呢

目前，real-vector检索已被证明可以提升QA系统的效果，但是real-vector是否可以有效的提升开放域对话系统仍然是一个待研究的问题。其次，目前的real-vector检索面临的主要问题就是存储空间大和查询速度相对慢的问题。我们也要提出一个基于Hash的vector语义检索模型，期望可以在不损失大量的效果的前提下，可以获得极快的查询速度和极低的存储大小，以促进检索式对话系统的实际应用和部署，比如移动设备上等（大量的内积运算消耗太多的能量和电力，使用哈希的方法可以极大的降低运算功率）。

## 1. How to Use
### 1.0 Init this repo

```bash
./run.sh init
```

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

### 1.4 Obtain the index storage of the Elasticsearch

```bash
curl -X GET localhost:9200/_cat/indices?
```

### 1.5 Prepare the Pre-constructed Corpus (ES or FAISS)

ES doesn't need the gpu_id (set as 0); FAISS need the gpu_ids (default set as 1,2,3,4)

```bash
./prepare_corpus.sh <dataset_name> <es/faiss> 1,2,3,4
```

### 1.6 Chat test

```bash
# set faiss_cuda as -1 to use cpu, set faiss_cuda i>=0 to use gpu(cuda:i)
./chat.sh <dataset_name> <es/dense/hash> <top-k> <cuda> <faiss_cuda>
```

## 2. Experiment Results
### 2.1 Comparsion between Term-Frequency and Dense vector retrieval
1. Compare the performance, the ratio of the unconditional responses, the storage, and the time cost (The pre-constructed corpus is the combination of the train and test dataset).
2. 在这里还需要分析倒排索引查询时间复杂度和内积矩阵运算的时间复杂度作为辅助说明。
3. Average Coherence scores are calculated by the cross-bert model.

<center> <b> E-Commerce Dataset 109105 utterances (xx.xx%); batch size is 32 </b> </center>

| Method       | Top-20 | Top-100 | Coherence-20 | Coherence-100 | Storage | Time Cost (20/100) |
| :----------: | :----: | :-----: | :----------: | :-----------: | :-----: | :----------------: |
| BM25         | 0.025  | 0.055   | 0.615        | 0.5122        | 8.8Mb   | 0.0895s/0.1294s    |
| Dense (cpu)  | 0.204  | 0.413   | 0.9537       | 0.9203        | 802Mb   | 0.3893s/0.4015s    |
| Dense (gpu)  | 0.204  | 0.413   | 0.9537       | 0.9203        | 802Mb   | 0.0406s/0.0398s    |

<center> <b> Douban Dataset 442280 utterances (xx.xx%); batch size is 32 </b> </center>

| Method       | Top-20 | Top-100 | Coherence-20 | Coherence-100 | Storage | Time Cost (20/100) |
| :----------: | :----: | :-----: | :----------: | :-----------: | :-----: | :----------------: |
| BM25         | 0.063  |  0.096  |              |               | 55.4Mb  | 0.4487s/0.4997s    |
| Dense (cpu)  | 0.054  |  0.1049 |              |               | 1.3Gb   | 1.6011s/1.6797s    |
| Dense (gpu)  | 0.054  |  0.1049 |              |               | 1.3Gb   | 0.2s/0.1771s       |

<center> <b> LCCC Dataset 1650881 utterances (xx.xx%); batch size is 32 </b> </center>

| Method       | Top-20 | Top-100 | Coherence-20 | Coherence-100 | Storage | Time Cost (20/100) |
| :----------: | :----: | :-----: | :----------: | :-----------: | :-----: | :----------------: |
| BM25         | 0.0378 | 0.0695  |              |               | 115.9Mb | 0.1941s/0.2421s    |
| Dense (cpu)  | 0.0278 | 0.057   |              |               | 4.8 Gb  | 0.48s/0.4869s      |
| Dense (gpu)  | 0.0278 | 0.057   |              |               | 4.8 Gb  | 0.4662s/0.4622s    |

**Conclusion:**

### 2.2 Comparsion between the Dense vector and Hash vector retrieval
Compare the performance and the time cost

<center> <b> E-Commerce Dataset </b> </center>

| Method | Top-20 | Top-100 | Average Time Cost (batch=32) | Storage |
| :----: | :----: | :-----: | :------------------: | :---: |
| Dense (cpu)  | 0.18   | 0.345   |   1.0119s            | |
| Dense (gpu)  |  |    |            | |
| Hash (cpu)  |   |    |              | |
| Hash (gpu)  |  |    |            | |

<center> <b> Douban Dataset </b> </center>

| Method | Top-20 | Top-100 | Average Time Cost (batch=32) | Storage |
| :----: | :----: | :-----: | :------------------: | :---: |
| Dense (cpu)  |    |    |               | |
| Dense (gpu)  |  |    |            | |
| Hash (cpu)  |   |    |              | |
| Hash (gpu)  |  |    |            | |

### 2.3 Overall comparsion
cross-bert post rank with different coarse retrieval strategies. 
Performance metric is not the Top-20/100, should be the human evaluation or the other automatic evaluation. Average Time cost is also needed.
Datasets are: E-Commerce, Douban, LCCC.

<center> <b> E-Commerce Dataset </b> </center>

| Method | Top-20 | Top-100 | Average Time Cost (batch=32) | Storage |
| :----: | :----: | :-----: | :------------------: | :----: |
| BM25+cross-bert  |    |    |               | |
| Dense(cpu)+cross-bert  |  |    |            | |
| Dense(gpu)+cross-bert  |  |    |            | |
| Hash(cpu)+cross-bert  |   |    |              | |
| Hash(gpu)+cross-bert  |  |    |            | |

<center> <b> Douban Dataset </b> </center>

| Method | Top-20 | Top-100 | Average Time Cost (batch=32) | Storage |
| :----: | :----: | :-----: | :------------------: | :---: |
| BM25+cross-bert  |    |    |               | |
| Dense(cpu)+cross-bert  |  |    |            | |
| Dense(gpu)+cross-bert  |  |    |            | |
| Hash(cpu)+cross-bert  |   |    |              | |
| Hash(gpu)+cross-bert  |  |    |            | |
    
    
## 3. Configure of the server and environment

* Hardware: 
    * 48 Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz
    * GPU GeForce GTX 1080 Ti
* System: Ubuntu 18.04
* faiss-gpu 1.6.3
* Elasticsearch 7.6.1 & Lucene 8.4.0 & elasticsearch-py 7.6.0