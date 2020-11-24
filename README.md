# HashRetrieval
The codes of the paper: Ultra-Fast and Low-Memory Open-Domain Retrieval Dialog Systems by Using Semantic Hashing

现有的检索式开放域对话系统都主要分为两个阶段，分别是[粗筛检索和精细排序](https://dl.acm.org/doi/10.1145/3394486.3403211)。粗筛阶段目前都是使用BM25为主的term frequency的检索方法从预构建的语料库中选取和上下文主题比较相似的作为候选回复，精细排序阶段使用语义匹配模型从所有的候选回复中选取最合适的一个。目前大家的研究方向都放在构建一个更加精准的精细排序模型，但是目前，随着以bert为主的预训练模型逐渐在各个数据集的各个指标上取得了目前最好的成绩，精细排序模型的性能提升变得非常的困难，这使得提升检索式对话系统的主要瓶颈局限于如何构建一个更好更快的粗筛检索模块上。

但是目前的开放域对话系统的粗筛检索模块中，大家基本上都是用的是基于term frequency的方法，这种方法会召回和上下文具有相同的词或者词组的回复，在QA等其他任务中这样的粗筛检索模块是有效的，这是因为具有和问题一样的词的答案极大概率就是包含有正确答案的那个。但是这在开放域对话系统中却并不一定，在开放域对话系统中，和上下文具有相同的词或者词组（主题）的句子未必就是最合适的恢复，这使得使用传统的粗筛检索模块并不能有效的选出最合适的候选回复，从而导致性能的下降，如下所示（仍然需要数据支撑这个观点）：

* 上下文：我最喜欢的看惊悚和恐怖电影了
* 回复：我喜欢看恐怖电影
* 潜在候选回复：你为什么喜欢看这种猎奇类型呢

目前，real-vector检索已被证明可以提升QA系统的效果，但是real-vector是否可以有效的提升开放域对话系统仍然是一个待研究的问题。其次，目前的real-vector检索面临的主要问题就是存储空间大和查询速度相对慢的问题。我们也要提出一个基于Hash的vector语义检索模型，期望可以在不损失大量的效果的前提下，可以获得极快的查询速度和极低的存储大小，以促进检索式对话系统的实际应用和部署，比如移动设备上等（大量的内积运算消耗太多的能量和电力，使用哈希的方法可以极大的降低运算功率）。

## 1. How to Use
### 1.1 Init this repo

```bash
./run.sh init
pip install -r requirements.txt
```

### 1.2 Prepare Dataset and Get Statistic

* Prepare the datasets
    1. Download the Preprocessed datasets by us from this [link (password 6mzz)](https://pan.baidu.com/s/1oCDx-s6JZiafIxPLxVi2sQ):
    
        The metadata of four datasets are shown
    
        |    Datasets    | Train  | Test    | Source |
        |:--------------:|:------:|:-------:|:------:|
        |   E-Commerce   | 500000 | 1000    | [Data](https://drive.google.com/file/d/154J-neBo20ABtSmJDvm7DK0eTuieAuvw/view)       |
        |     Douban     | 500000 | 667     | [Data](https://github.com/MarkWuNLP/MultiTurnResponseSelection) |
        |      Zh50w     | 994002 | 2998    | [Data](https://github.com/yangjianxin1/GPT2-chitchat)       |
        | LCCC (partial) | 2000000| 10000   | [Data](https://github.com/thu-coai/CDial-GPT)       |

        _Note: Original Douban Multi-turn Datasets contains 1000 sessions in test split, but only 667 have the positive samples in it (legal)._
        
    2. Unzip the the zipped file:
        ```bash
        tar -xzvf hashretrieval_datasets.tgz
        ```
    3. Copy the `<dataset_name>/train.txt` and `<dataset_name>/test.txt` files to the corresponding dataset folder under `data`.

* Get the statistic of these datasets
```bash
./run.sh statistic
```

### 1.3 Train the dual-bert or hash-bert model

* dual-bert or hash-bert has two bert models, the batch size is 16
* **dataset_name: ecommerce, douban, zh50w, lccc**

```bash
# for example: ./run.sh train ecommerce dual-bert 0,1,2,3
./run.sh train <dataset_name> <dual-bert/hash-bert> <gpu_ids>
```

### 1.4 Train the cross-bert model

* cross-bert has one bert mode, the batch size is 32.
* E-Commerce and Douban datasets use 5e-5 learning ratio, but zh50w and lccc datasets need 2e-5 learning ratio (5e-5 not converge).

```bash
./run.sh train <dataset_name> cross-bert <gpu_ids>
```

### 1.5 Obtain the Index Storage of the Elasticsearch

```bash
curl -X GET localhost:9200/_cat/indices?
```

_Note: The storage information showed by the above command is the combination of the inverted index and the corpus. In order to obtain the storage of the inverted index, you need to minus the size of the corpus, which can be obtained by using this command:_

```bash
ls -lh data/<dataset_name>/dual-bert.corpus_ckpt
```

### 1.6 Prepare the Pre-constructed Corpus (es or faiss)

es doesn't need the gpu_id (set as 0); faiss needs the gpu_ids (default set as 1,2,3,4)

_If you need to try other hash code size settings, replace the 128 in `chat.sh` and `models/hash-bert.py` into other dimension size._

```bash
./prepare_corpus.sh <dataset_name> <es/faiss> <es/dual-bert/hash-bert> 1,2,3,4
```

### 1.7 Run Chat Test to obtain the Experiment Results

* If you need to try other hash code size settings, replace the 128 in `chat.sh` and `models/hash-bert.py`  into other dimension size.
* If you need to change the number of the negative samples, just replace the batch size in `run.sh` with the number that you want.
* Before running the chat.sh script, you should make sure that you already run the following commands correctly:
    ```bash
    # 1. the cross-bert will be used for providing coherence scores
    ./run.sh train <dataset_name> cross-bert <gpu_ids>
    # 2. the hash-bert model also needs the dual-bert model parameters
    ./run.sh train <dataset_name> dual-bert <gpu_ids>
    # 3. generate the embedding for the pre-constructed corpus
    ./prepare_corpus.sh <dataset_name> <es/faiss> <es/dual-bert/hash-bert> 0,1,2,3
    ```
* Test
    ```bash
    # set faiss_cuda as -1 to use cpu, set faiss_cuda i>=0 to use gpu(cuda:i)
    ./chat.sh <dataset_name> <es/dense/hash> <top-k> <cuda> <faiss_cuda>
    ```
    The generated responses will be saved under `generated/<dataset_name>/<es/dense/hash>`

### 1.8 Sample Results for Annotation

If you want to annotate and reproduce the results in Section 2.3, you can run the following commands to obtain the sampled corpus for annotation:

```bash
./export_corpus.sh
```

Then you can find the sampled files under four `generated/<dataset_name>` folders.

## 2. Experiment Results
1. the number of the utterances in the pre-constructed database, the ratio of the unconditional responses, the storage (only consider the faiss index and elasticsearch inverted index), and the search time cost.
2. Search Time Complexity (n is the number of the queries, m is the dimension of the real-vector or binary-vector):
    * Inverted Index: O(n)
    * Dot production: O(n*m)
    * Hamming Distance: O(n)
3. Average Coherence scores are calculated by the cross-bert model.
    
### 2.1 Comparsion between Term-Frequency and Dense vector retrieval

<center> <b> E-Commerce Dataset 109105 utterances (46.81%) </b> </center>

| Method       | Top-20 | Top-100 | Coherence-20 | Coherence-100 | Storage | Time Cost (20/100) |
| :----------: | :----: | :-----: | :----------: | :-----------: | :-----: | :----------------: |
| BM25         | 0.025  | 0.055   | 0.615        | 0.5122        | **2.9 Mb**  | 0.0895s/0.1294s    |
| Dense (cpu)  | 0.204  | 0.413   | 0.9537       | 0.9203        | 320 Mb  | 0.3893s/0.4015s    |
| Dense (gpu)  | **0.204**  | **0.413**   | **0.9537**       | **0.9203**        | 320 Mb  | **0.0406s/0.0398s**    |

<center> <b> Douban Dataset 442280 utterances (54.47%) </b> </center>

| Method       | Top-20 | Top-100 | Coherence-20 | Coherence-100 | Storage | Time Cost (20/100) |
| :----------: | :----: | :-----: | :----------: | :-----------: | :-----: | :----------------: |
| BM25         | **0.063**  |  0.096  | 0.6957       | 0.6057        | **21.4 Mb** | 0.4487s/0.4997s    |
| Dense (cpu)  | 0.054  |  0.1049 | 0.9403       | 0.9067        | 1.3 Gb  | 1.6011s/1.6797s    |
| Dense (gpu)  | 0.054  |  **0.1049** | **0.9403**       | **0.9067**        | 1.3 Gb  | **0.2s/0.1771s**       |

<center> <b> Zh50w Dataset 388614 utterances (28.5%) </b> </center>

| Method       | Top-20 | Top-100 | Coherence-20 | Coherence-100 | Storage | Time Cost (20/100) |
| :----------: | :----: | :-----: | :----------: | :-----------: | :-----: | :----------------: |
| BM25         | **0.0627** | **0.1031**  | 0.84         | 0.7341        | **10.8 Mb** | **0.0915s/0.1228s**    |
| Dense (cpu)  | 0.044  | 0.0824  | 0.9655       | 0.9424        | 1.2 Gb  | 0.1137s/0.127s     |
| Dense (gpu)  | 0.044  | 0.0824  | **0.9655**       | **0.9424**        | 1.2 Gb  | 0.1224s/0.1283s    |

<center> <b> LCCC Dataset 1651899 utterances (33.59%) </b> </center>

| Method       | Top-20 | Top-100 | Coherence-20 | Coherence-100 | Storage | Time Cost (20/100) |
| :----------: | :----: | :-----: | :----------: | :-----------: | :-----: | :----------------: |
| BM25         | **0.0376** | 0.07    | 0.8966       | 0.8253        | **44 Mb**   | **0.1901s/0.247s**     |
| Dense (cpu)  | 0.0351 | 0.0778  | 0.9832       | 0.9726        | 4.8 Gb  |    0.592s/0.6685s |
| Dense (gpu)  | 0.0351 | **0.0778**  | **0.9832**       | **0.9726**        | 4.8 Gb  | 0.4586s/0.5722s    |

**Conclusion:**
* Dense检索方法与传统的BM25检索方法比，在召回的文本的整体的相关性上具有巨大的优势，这里我们并不看重Top-20/100指标的效果是因为，Top-20/100无法翻译召回的所有候选文本的整体情况
* 使用GPU加速的Dense实值向量检索方法速度明显提升
* 因为需要存储大量的实值向量作为索引的键值，Dense方法的存储空间远大于BM25的倒排索引，并且查询速度并不如BM25方法快

### 2.2 Comparsion between Dense vector and Hash vector retrieval
* Storage is the size of inverted index or the vector index.
* default hash code size is 512/128.
* default hash-bert batch size is 16.

<center> <b> E-Commerce Dataset 109105 utterances (46.81%) </b> </center>

| Method       | Top-20 | Top-100 | Coherence-20 | Coherence-100 | Storage | Time Cost (20/100) |
| :----------: | :----: | :-----: | :----------: | :-----------: | :-----: | :----------------: |
| BM25         | 0.025  | 0.055   | 0.615        | 0.5122        | 2.9 Mb  | 0.0895s/0.1294s    |
| Dense (gpu)  | 0.204  | **0.413**   | **0.9537**       | **0.9203**        | 320 Mb  | 0.3893s/0.4015s    |
| Hash-128 (gpu)  |  0.185  | 0.366 | 0.9252  | 0.8808  | **1.7 Mb** | **0.004s/0.0063s** | 
| Hash-512 (gpu)  | **0.214**  | 0.382   | 0.944        | 0.9065        | 6.7 Mb  | 0.0093s/0.0187s    |  

<center> <b> Douban Dataset 442280 utterances (54.47%) </b> </center>

| Method       | Top-20 | Top-100 | Coherence-20 | Coherence-100 | Storage | Time Cost (20/100) |
| :----------: | :----: | :-----: | :----------: | :-----------: | :-----: | :----------------: |
| BM25         | **0.063** |  0.096  | 0.6957       | 0.6057        | 21.4 Mb | 0.4487s/0.4997s    |
| Dense (gpu)  | 0.054  | **0.1049**  | **0.9403**       | **0.9067**        | 1.3 Gb  | 0.2s/0.1771s       |
| Hash-128 (gpu)  |  0.012 | 0.0465  |  0.8375      |  0.8016       | **6.8 Mb**  | **0.0209s/0.0196s**    | 
| Hash-512 (gpu)  | 0.0225 | 0.066   | 0.8838       | 0.8474        | 27 Mb   | 0.0523s/0.0452s    |

<center> <b> Zh50w Dataset 388614 utterances (28.5%) </b> </center>

| Method       | Top-20 | Top-100 | Coherence-20 | Coherence-100 | Storage | Time Cost (20/100) |
| :----------: | :----: | :-----: | :----------: | :-----------: | :-----: | :----------------: |
| BM25         | **0.0627** | **0.1031**  | 0.84         | 0.7341        | **10.8 Mb** | 0.0915s/0.1228s    |
| Dense (gpu)  | 0.044  | 0.0824  | **0.9655**       | **0.9424**        | 1.2 Gb  | 0.1224s/0.1283s    |
| Hash-128 (gpu)  |    |  |   |   |  |  | 
| Hash-512 (gpu)  | 0.0377 | 0.0934  | 0.944        | 0.9223        | 24 Mb   | **0.0235s/0.028s**     |

<center> <b> LCCC Dataset 1651899 utterances (33.59%) </b> </center>

| Method       | Top-20 | Top-100 | Coherence-20 | Coherence-100 | Storage | Time Cost (20/100) |
| :----------: | :----: | :-----: | :----------: | :-----------: | :-----: | :----------------: |
| BM25         | **0.0376** | 0.07    | 0.8966       | 0.8253        | **44 Mb**   | 0.1901s/0.247s     |
| Dense (gpu)  | 0.0351 | **0.0778**  | **0.9832**       | **0.9726**        | 4.8 Gb  | 0.4586s/0.5722s    |
| Hash-128 (gpu)  |    |  |   |   |  |  | 
| Hash-512 (gpu)  | 0.0204 | 0.0494  | 0.9663       | 0.9526        | 101 Mb  | **0.0764s/0.094s**     |

**Conclusion:**
* 使用了哈希的方法之后，可以发现仅仅损失了相当少的性能损失，但是我们得到了极低的存储空间和几块的查询速度
* 哈希方法相比于传统的BM25方法，保留了语义的相似度的极高的查询相关性
* 虽然存储空间比BM25略大一点，但是注意我们使用的是512维哈希码存储，实际上128维的哈希码已经可以得到很好的效果同时128维的哈希码存储空间比BM25要小，即使是16维的哈希码，效果依然比BM25方法好，同时存储空间最小。

### 2.3 Overall Comparsion (Coarse retrieval + Cross-bert Post Rank)

Human Evaluation
* Each dataset have 500 samples to be annotated
* 3 annotators are used

<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt" rowspan="2">Models</th>
    <th class="tg-7btt" colspan="3">E-Commerce</th>
    <th class="tg-7btt" colspan="3">Douban</th>
    <th class="tg-7btt" colspan="3">Zh50w</th>
    <th class="tg-7btt" colspan="3">LCCC</th>
  </tr>
  <tr>
    <td class="tg-7btt">Win</td>
    <td class="tg-7btt">Loss</td>
    <td class="tg-7btt">Tie</td>
    <td class="tg-7btt">Win</td>
    <td class="tg-7btt">Loss</td>
    <td class="tg-7btt">Tie</td>
    <td class="tg-7btt">Win</td>
    <td class="tg-7btt">Loss</td>
    <td class="tg-7btt">Tie</td>
    <td class="tg-7btt">Win</td>
    <td class="tg-7btt">Loss</td>
    <td class="tg-7btt">Tie</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7btt">BM25</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
  </tr>
  <tr>
    <td class="tg-7btt">Dense</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
  </tr>
  <tr>
    <td class="tg-7btt">Hash</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
  </tr>
</tbody>
</table>

**Conclusion:**

### 2.4 Hyperparameters

#### 2.4.1 Hash code size

_Note: Default the Number of Negative Samples is 16_

<center> <b> E-Commerce Dataset 109105 utterances (46.81%) </b> </center>

| Method        | Coherence-20 | Coherence-100 | Storage | Time Cost (20/100) |
| :-----------: | :----------: | :-----------: | :-----: | :----------------: |
| BM25          | 0.615        | 0.5122        | 8.8 Mb  | 0.0895s/0.1294s    |
| Hash-16 (gpu) | 0.6819       | 0.6284        | **214 Kb**  | **0.0026s/0.0051s**    |
| Hash-32 (gpu) | 0.8271       | 0.7714        | 427 Kb  | 0.0017s/0.0031s    |
| Hash-48 (gpu) | 0.8737       | 0.8136        | 640 Kb  | 0.0064s/0.0081s    |
| Hash-64 (gpu) | 0.8942       | 0.8364        | 853 Kb  | 0.0018s/0.0044s    |
| Hash-128 (gpu)| 0.9278       | 0.8837        | 1.7 Mb  | 0.0023s/0.0044s    |
| Hash-256 (gpu)| 0.9376       | 0.8976        | 3.4 Mb  | 0.0032s/0.0106s    |
| Hash-512 (gpu)| 0.944        | 0.9065        | 6.7 Mb  | 0.0092s/0.0164s    |
|Hash-1024 (gpu)| **0.9473**   | **0.9134**    | 14 Mb   | 0.0194s/0.0184s    |
| Dense (gpu)   | **0.9537**   | **0.9203**    | 320 Mb  | 0.3893s/0.4015s    |

<center> <b> Zh50w Dataset 388614 utterances (28.5%) </b> </center>

| Method        | Coherence-20 | Coherence-100 | Storage | Time Cost (20/100) |
| :-----------: | :----------: | :-----------: | :-----: | :----------------: |
| BM25          | 0.84         | 0.7341        | 10.8 Mb | 0.0915s/0.1228s    |
| Hash-16 (gpu) | 0.6703       | 0.6431        | **760 Kb** | 0.0093s/0.0163s    |
| Hash-32 (gpu) | 0.7912       | 0.7557        | 1.5 Mb  | **0.005s/0.0065s**     |
| Hash-48 (gpu) | 0.8428       | 0.8101        | 2.3 Mb  | 0.0193s/0.0207s    |
| Hash-64 (gpu) | 0.8685       | 0.836         | 3.0 Mb  | 0.0051s/0.0063s    |
| Hash-128 (gpu)| 0.9108       | 0.8835        | 6.0 Mb  | 0.0094s/0.0174s    |
| Hash-256 (gpu)| 0.9353       | 0.9105        | 12 Mb   | 0.016s/0.0133s     |
| Hash-512 (gpu)| 0.944        | 0.9223        | 24 Mb   | 0.0235s/0.028s     |
|Hash-1024 (gpu)| 0.9546       | 0.9336        | 48 Mb   | 0.0502s/0.0647s    |
| Dense (gpu)   | **0.9655**   | **0.9424**    | 1.2 Gb  | 0.1224s/0.1283s    |

#### 2.5.1 The Number of the Negative Samples

_Note: Default Hash Code Size is 512_

<center> <b> E-Commerce Dataset 109105 utterances (46.81%) </b> </center>

| Method (Negative Samples) | Coherence-20 | Coherence-100 | Storage | Time Cost (20/100) |
| :-----------: | :----------: | :-----------: | :-----: | :----------------: |
| BM25          | 0.615        | 0.5122        | 8.8 Mb  | 0.0895s/0.1294s    |
| 16 (gpu) | **0.944**        | **0.9065**        | 6.7 Mb  | 0.0092s/0.0164s    |
| 32 (gpu) | 0.9412       | 0.9041        | 6.7 Mb  | 0.0084s/0.0134s    |
| 64 (gpu) | 0.9425       | 0.902         | 6.7 Mb  | 0.0089s/0.0122s    |
| 128 (gpu)| 0.9398       | 0.8974        | 6.7 Mb  | 0.0091s/0.0108s    |
| Dense (gpu)   | **0.9537**       | **0.9203**        | 320 Mb  | 0.3893s/0.4015s    |

**Conclusion:**
* The bigger negative samples, worse performance. Best negative samples is 16.
* The bigger hash code size, the higher performance. Best hash code size is 512 (1024).

## 3. Case Study

### 3.1 Coarse Retrieval Case Study

### 3.2 Overall Case Study

#### 3.2.1 E-Commerce Dataset
* Case 1:
    1. Context: 这快递咋回事啊有史以来最慢的一次
    2. Reference: 您好
    3. Dense: 不好意思亲最近EMS到广东的件都有点慢抱歉呢这边给亲催下
    4. Hash: 您好不好意思这边快递下班了帮您留言催下
    5. ES: 这是最慢的估算哦
    
* Case 2:
    1. Context: 我今天买什么时候到什么时候发货 [SEP] 您好亲亲拍下后35天发货哦 [SEP] 这么晚呀哎
    2. Reference: 恩呢
    3. Dense: 亲亲我们拍下后35天发货的哦实在抱歉呢过年前订单量大
    4. Hash: 亲亲我们是拍下35天发货哦
    5. ES: 什么时候买的呢
    
* Case 3:
    1. Context: 亲在线吗 [SEP] 在的呢 [SEP] 为什么还不发货
    2. Reference: 客官榴莲干刚刚到货哦今明两天给您发货哦
    3. Dense: 发货了呢
    4. Hash: 实在抱歉呢大号红茶刀暂时缺货了哦估计要明后天才能发货了呢
    5. ES: 已经出库了哦亲为什么退呢
    
* Case 4:
    1. Context: 好的 [SEP] 您拍下给您改价 [SEP] 多久到新疆
    2. Reference: 亲爱哒客官小店是从浙江嘉兴发货的哦一般发货后4-6天就能到您那边呢请您耐心等下哦
    3. Dense: 大概23天哦
    4. Hash: 发货后45tian天
    5. ES: 您拍下我给您改价哦
    
* Case 5:
    1. Context: 拍了尽量早点发谢谢 [SEP] 亲请核对一下收货地址哦 [SEP] 对的
    2. Reference: 好的小店尽快给您发出哦
    3. Dense: 好的小店明天给您发出哦
    4. Hash: 好的小店明天给您发出哦
    5. ES: 小二这边尽量给您快点发出哦亲请核对一下收货地址哦
    
* Case 6:
    1. Context: 我买的洗衣皂怎么到了长沙又转其他地方了
    2. Reference: 您好这边联系快递核实下
    3. Dense: 您好帮亲联系下快递哦
    4. Hash: 您好应该是中转这边联系快递催下吧
    5. ES: 怎么可能又不是我转
    
* Case 7:
    1. Context: 现货没有吗 [SEP] 有的哦订单量大实在抱歉呢 [SEP] 核桃质量咋样
    2. Reference: 放心的哦亲亲
    3. Dense: 挺不错哦
    4. Hash: 都是新货哦我们不做陈货的呢奶香味足
    5. ES: 订单量大实在抱歉让您久等了
    
* Case 8:
    1. Context: 这个近期有优惠吗 [SEP] 本月16号有活动哦亲到时可以关注下呢 [SEP] 好的
    2. Reference: 嗯嗯亲
    3. Dense: 亲到时再关注下哦
    4. Hash: 嗯嗯亲到时可以关注下呢
    5. ES: 本月16号有周年庆的活动亲到时可以关注下的
    
* Case 9:
    1. Context: 亲己经拍好了 [SEP] 好的 [SEP] 送点好的小吃给我尝一下 [SEP] 好的 [SEP] 好的话以后就选你家 [SEP] 好的呢亲 [SEP] 我是牺牲品多让我试吃 [SEP] 哈哈 [SEP] 我快要变成吃货了 [SEP] 哈哈一直都是吃货吧 [SEP] 所以说要把我这个吃货要哄好哦
    2. Reference: 好的呢
    3. Dense: 好的哈
    4. Hash: 好的呢呢亲
    5. ES: 一看您就是个吃货哈哈哈

* Case 10:
    1. Context: 哦哦日期新鲜吗现在还能发货吗 [SEP] 明后天发货哦 [SEP] 日期多少 [SEP] 33 [SEP] 用什么快递发货 [SEP] 亲爱哒客官小店默认邮政百世汇通天天快递哦 [SEP] 百世汇通 [SEP] 好的 [SEP] 付款了
    2. Reference: 好的
    3. Dense: 亲请核对一下收货地址哦今天给您发出哦
    4. Hash: 亲请核对一下收货地址哦今天给您发出哦
    5. ES: 好的亲爱哒客官小店默认邮政百世汇通天天快递哦

#### 3.2.2 Douban Dataset

#### 3.2.3 Zh50w Dataset

#### 3.2.4 LCCC Dataset
* Case 1:
    1. Context: 喵的我都想转发了，救命这简直太好看了
    2. Reference: 你也超好看的好吗！看到你我都不敢抬头了
    3. Dense: 啊啊啊啊啊啊，想和全世界安利我的猫。
    4. Hash: 实在是太好看了呜呜呜
    5. ES: 救命啊我好委屈
    
* Case 2:
    1. Context: 易烊千玺 [SEP] 这张吼吼看啊
    2. Reference: 千玺那张不好看
    3. Dense: 超超超超超级好看了
    4. Hash: 特别特别好看呀
    5. ES: 必须易烊千玺啊
    
* Case 3:
    1. Context: 喘不过气。旁人还要觉得你矫情。 [SEP] 懂的，超难受
    2. Reference: 别说了。很想死了。
    3. Dense: 难受了找贴心的人说说聊聊，诉苦又不丢人。
    4. Hash: 难受了找贴心的人说说聊聊，诉苦又不丢人。
    5. ES: 你矫情啥。
    
* Case 4:
    1. Context: 我也有心理阴影了可能不会尝试第二次了 [SEP] 我们吃别的！
    2. Reference: 好好好吃别的好吃哒
    3. Dense: 你吃的话那我们吃什么
    4. Hash: 你吃的啥给我点建议呗
    5. ES: 我可能这辈子不会再尝试第二次
    
* Case 5:
    1. Context: 像我这么有智慧的人一般都没人知道 [SEP] 你女朋友需要吗
    2. Reference: 你说的啥我不知道
    3. Dense: 好像挺需要的
    4. Hash: 根本不需要
    5. ES: 不要自恋，所以你是真的有智慧
    
* Case 6:
    1. Context: 内个女的很厉害！！
    2. Reference: 这教育了我们不要随便领养小孩我的那个天啊伤不起啊真的最后那个翻转吓到我了
    3. Dense: 我也觉得超厉害！一个女生！就更厉害了！
    4. Hash: 我也觉得超厉害！一个女生！就更厉害了！
    5. ES: 内个治百病
    
* Case 7:
    1. Context: 今天真的帅呆了 [SEP] 问一下，这是在哪？
    2. Reference: 上海宝马嘉年华活动
    3. Dense: 就在网易背后小公园背后的地下广场里
    4. Hash: 在西安外事学院南区门口对面
    5. ES: 问一下，这是在哪？
    
* Case 8:
    1. Context: 好吃么泰国美食 [SEP] 好吃！不过要根据攻略点，不然就是黑暗料理
    2. Reference: 可以把你攻略发我一份么
    3. Dense: 哈哈谢谢啦昨晚看了攻略啦明天就去
    4. Hash: 你写的游记我很喜欢，作为一枚吃货，你介绍的美食我都去尝了，真的很好吃
    5. ES: 无需要你的黑暗料理
    
* Case 9:
    1. Context: 我也去过这个青龙镇，今年新开发的 [SEP] 还没收费
    2. Reference: 我去的时候也是，刚刚开发出来，还没有收费
    3. Dense: 怎么可能免费
    4. Hash: 免费的就没价值了
    5. ES: 我收费很贵的
    
* Case 10:
    1. Context: 我现在看剧看到枯竭了推荐你个电影水果硬糖。特好看
    2. Reference: 我看过了
    3. Dense: 看过了，超甜，牙都甜掉了
    4. Hash: 看过了，为什么要推荐这部？
    5. ES: 不行了，枯竭了
    
**Conclusion:**
1. 使用real-vector检索(Dense/Hash)往往召回的都是和上下文语义相关的回复，这一点上传统的BM25的检索方法都会导致很差的结果，比如利用错误的term召回回复或者回复和上下文语义根本不相近
2. 

## 4. Test Configuration

* Hardware: 
    * 48 Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz
    * GPU GeForce GTX 1080 Ti
* System: Ubuntu 18.04
* Software:
    * python 3.6
    * PyTorch 1.5-cu92
    * Huggingface Transformers 2.11.0
    * faiss-gpu 1.6.3
    * Elasticsearch 7.6.1 & Lucene 8.4.0 & elasticsearch-py 7.6.0

## 5. Future works

* Multiple heads hash module, which can save more semantic embedding information.
* Combine the advantanges of BM25 and dense vector. A vector retrieval model which is sensitive to the keywords overlap. 例如考虑关键词的哈希相似度函数
* Coherence-20/100和Top-20/100之间的差别说清楚，为什么用coherence不用top要说清，说服别人
* 统计一下哈希码存储空间和语料存储空间的大小对比，凸显出哈希码对比于dense检索的优势
* 现在的检索式对话系统依然不grounding，也是盲目的匹配最近语义的句子，比如在电商领域，发货时间信息等这些必须要有真是的数据作为支撑，不能无脑的选择最合适的句子作为回复，这样的回复里面的信息完全不grounding。我觉得解决这个问题的方法之一可以是加入更多的信息（比如这些grounding的信息）作为context。
* Better deep hash representation model should be used