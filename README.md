# HashRetrieval
The codes of the paper: Ultra-Fast and Low-Memory Open-Domain Retrieval Dialog Systems by Using Semantic Hashing

## 1. How to Use
### 1.1 Init this repo

```bash
./run.sh init
pip install -r requirements.txt
```

### 1.2 Prepare Dataset and Get Statistic

* Prepare the datasets
    1. Download the Preprocessed datasets by us from this [link (password 6mzz)](https://pan.baidu.com/s/1oCDx-s6JZiafIxPLxVi2sQ):
    
        The metadata of four datasets are shown as follows:
    
        |    Datasets    | Train  | Test    | Coherence Ratio | Source |
        |:--------------:|:------:|:-------:| :--------------:|:------:|
        |   E-Commerce   | 500000 | 1000    | 46.81%          | [Data](https://drive.google.com/file/d/154J-neBo20ABtSmJDvm7DK0eTuieAuvw/view)       |
        |     Douban     | 500000 | 667     | 54.57%          | [Data](https://github.com/MarkWuNLP/MultiTurnResponseSelection) |
        |      Zh50w     | 994002 | 2998    | 28.5%           |[Data](https://github.com/yangjianxin1/GPT2-chitchat)       |
        | LCCC (partial) | 2000000| 10000   | 33.59%          |[Data](https://github.com/thu-coai/CDial-GPT)       |

        _Note: 1. Original Douban Multi-turn Datasets contains 1000 sessions in test split, but only 667 have the positive samples in it (legal); 2. Coherence ratoi represents the ratio of the samples that the responses can be found by the context._
        
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
| BM25         | **0.0627** | **0.1031**  | 0.84         | 0.7341        | 10.8 Mb | 0.0915s/0.1228s    |
| Dense (gpu)  | 0.044  | 0.0824  | **0.9655**       | **0.9424**        | 1.2 Gb  | 0.1224s/0.1283s    |
| Hash-128 (gpu)  |  0.027  | 0.0724 |  0.9108 | 0.8835  | **6.0 Mb** | **0.0137s/0.0192s** | 
| Hash-512 (gpu)  | 0.0377 | 0.0934  | 0.944        | 0.9223        | 24 Mb   | 0.0235s/0.028s     |

<center> <b> LCCC Dataset 1651899 utterances (33.59%) </b> </center>

| Method       | Top-20 | Top-100 | Coherence-20 | Coherence-100 | Storage | Time Cost (20/100) |
| :----------: | :----: | :-----: | :----------: | :-----------: | :-----: | :----------------: |
| BM25         | **0.0376** | 0.07    | 0.8966       | 0.8253        | 44 Mb   | 0.1901s/0.247s     |
| Dense (gpu)  | 0.0351 | **0.0778**  | **0.9832**       | **0.9726**        | 4.8 Gb  | 0.4586s/0.5722s    |
| Hash-128 (gpu)  | 0.014  | 0.0348  | 0.9369       | 0.9187        | **26 Mb** | **0.0204s/0.0244s** | 
| Hash-512 (gpu)  | 0.0204 | 0.0494  | 0.9663       | 0.9526        | 101 Mb  | 0.0764s/0.094s     |

**Conclusion:**
* 使用了哈希的方法之后，可以发现仅仅损失了相当少的性能损失，但是我们得到了极低的存储空间和几块的查询速度
* 哈希方法相比于传统的BM25方法，保留了语义的相似度的极高的查询相关性
* 虽然存储空间比BM25略大一点，但是注意我们使用的是512维哈希码存储，实际上128维的哈希码已经可以得到很好的效果同时128维的哈希码存储空间比BM25要小，即使是16维的哈希码，效果依然比BM25方法好，同时存储空间最小。

### 2.3 Overall Comparsion (Coarse retrieval + Cross-bert Post Rank)

[Human Evaluation (three labels classification)](https://arxiv.org/abs/2009.07543)
* Each dataset have 200 samples to be annotated
* 3 annotators are used
* Top-100 coarse retrieval module and the cross-bert post rank model are used to generate the test responses.

<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt" rowspan="2">Models</th>
    <th class="tg-7btt" colspan="4">E-Commerce</th>
    <th class="tg-7btt" colspan="4">Douban</th>
    <th class="tg-7btt" colspan="4">Zh50w</th>
    <th class="tg-7btt" colspan="4">LCCC</th>
  </tr>
  <tr>
    <td class="tg-7btt">Win</td>
    <td class="tg-7btt">Loss</td>
    <td class="tg-7btt">Tie</td>
    <td class="tg-7btt">Kappa</td>
    <td class="tg-7btt">Win</td>
    <td class="tg-7btt">Loss</td>
    <td class="tg-7btt">Tie</td>
    <td class="tg-7btt">Kappa</td>
    <td class="tg-7btt">Win</td>
    <td class="tg-7btt">Loss</td>
    <td class="tg-7btt">Tie</td>
    <td class="tg-7btt">Kappa</td>
    <td class="tg-7btt">Win</td>
    <td class="tg-7btt">Loss</td>
    <td class="tg-7btt">Tie</td>
    <td class="tg-7btt">Kappa</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7btt">Dense vs. BM25</td>
    <td class="tg-c3ow">0.5917</td>
    <td class="tg-c3ow">0.2117</td>
    <td class="tg-c3ow">0.1967</td>
    <td class="tg-c3ow">0.7679</td>
    <td class="tg-c3ow"><b>0.4783</b></td>
      <td class="tg-c3ow"><b>0.1883</b></td>
    <td class="tg-c3ow">0.3333</td>
    <td class="tg-c3ow">0.8240</td>
      <td class="tg-c3ow"><b>0.5017</b></td>
      <td class="tg-c3ow"><b>0.2683</b></td>
    <td class="tg-c3ow">0.23</td>
    <td class="tg-c3ow">0.7143</td>
    <td class="tg-c3ow">0.5233</td>
    <td class="tg-c3ow">0.305</td>
    <td class="tg-c3ow">0.1717</td>
    <td class="tg-c3ow">0.5558</td>
  </tr>
  <tr>
    <td class="tg-7btt">Hash vs. BM25</td>
    <td class="tg-c3ow"><b>0.6017</b></td>
    <td class="tg-c3ow"><b>0.1933</b></td>
    <td class="tg-c3ow">0.205</td>
    <td class="tg-c3ow">0.6733</td>
    <td class="tg-c3ow">0.4767</td>
    <td class="tg-c3ow">0.2783</td>
    <td class="tg-c3ow">0.245</td>
    <td class="tg-c3ow">0.8506</td>
    <td class="tg-c3ow">0.4733</td>
    <td class="tg-c3ow">0.335</td>
    <td class="tg-c3ow">0.1917</td>
    <td class="tg-c3ow">0.727</td>
      <td class="tg-c3ow"><b>0.5317</b></td>
      <td class="tg-c3ow"><b>0.27</b></td>
    <td class="tg-c3ow">0.1983</td>
    <td class="tg-c3ow">0.7115</td>
  </tr>
  <tr>
    <td class="tg-7btt">Dense vs. Hash</td>
    <td class="tg-c3ow">0.3133</td>
    <td class="tg-c3ow">0.2683</td>
    <td class="tg-c3ow"><b>0.4183</b></td>
    <td class="tg-c3ow">0.7025</td>
    <td class="tg-c3ow">0.375</td>
    <td class="tg-c3ow">0.2217</td>
      <td class="tg-c3ow"><b>0.4033</b></td>
    <td class="tg-c3ow">0.8251</td>
    <td class="tg-c3ow">0.395</td>
    <td class="tg-c3ow">0.2833</td>
      <td class="tg-c3ow"><b>0.3217</b></td>
    <td class="tg-c3ow">0.6679</td>
    <td class="tg-c3ow">0.3283</td>
    <td class="tg-c3ow">0.3733</td>
      <td class="tg-c3ow"><b>0.2983</b></td>
    <td class="tg-c3ow">0.7716</td>
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

<center> <b> Douban Dataset 109105 utterances (54.47%) </b> </center>

| Method        | Coherence-20 | Coherence-100 | Storage | Time Cost (20/100) |
| :-----------: | :----------: | :-----------: | :-----: | :----------------: |
| BM25          | 0.6957       | 0.6057        | 21.4 Mb | 0.4487s/0.4997s    |
| Hash-16 (gpu) |              |               |         |                    |
| Hash-32 (gpu) |              |               |         |                    |
| Hash-48 (gpu) |              |               |         |                    |
| Hash-64 (gpu) |              |               |         |                    |
| Hash-128 (gpu)| 0.8375       | 0.8016        | 6.8 Mb  | 0.0209s/0.0196s    |
| Hash-256 (gpu)|              |               |         |                    |
| Hash-512 (gpu)| 0.8838       | 0.8474        | 27 Mb   | 0.0523s/0.0452s    |
|Hash-1024 (gpu)|              |               |         |                    |
| Dense (gpu)   | 0.9403       | 0.9067        | 1.3 Gb  | 0.2s/0.1771s       |

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

<center> <b> LCCC Dataset 109105 utterances (33.59%) </b> </center>

| Method        | Coherence-20 | Coherence-100 | Storage | Time Cost (20/100) |
| :-----------: | :----------: | :-----------: | :-----: | :----------------: |
| BM25          | 0.8966       | 0.8253        |  44 Mb  | 0.1901s/0.247s     |
| Hash-16 (gpu) |              |               |         |                    |
| Hash-32 (gpu) |              |               |         |                    |
| Hash-48 (gpu) |              |               |         |                    |
| Hash-64 (gpu) |              |               |         |                    |
| Hash-128 (gpu)| 0.9369       | 0.9187        |  26 Mb  | 0.0204s/0.0244s    |
| Hash-256 (gpu)|              |               |         |                    |
| Hash-512 (gpu)| 0.9663       | 0.9526        |  101 Mb | 0.0764s/0.094s     |
|Hash-1024 (gpu)|              |               |         |                    |
| Dense (gpu)   | 0.9832       | 0.9726        |  4.8 Gb | 0.4586s/0.5722s    |

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

* Case 1:
    1. Context: 我想问问郑州哪里有订做衣服的地方 [SEP] 城东路商都美术管对面皮料市场那边可多定做皮件的 [SEP] 我就住附近我都不知道耶 [SEP] 咦咱俩住好近
    2. Reference: 真近是多近啊童鞋
    3. Dense: 好近什么名字或者具体点位置
    4. Hash: 哈哈哈那是近得很耶在岗顶上班吗
    5. ES: 我去我就是郑州的
    
* Case 2:
    1. Conetxt: 痔疗对策 [SEP] 啊啊啊每天便便顺畅可是还是得了痔疮啊 [SEP] 同说多了都是泪啊 [SEP] 腹泻啊啊啊 [SEP] 我也是啊啊啊啊啊 [SEP] 养好肠胃是关键啊
    2. Reference: 默默的握手发现两个同病相怜的人我也是肠胃一直不好所以最近都不吃辣椒总是嚼口香糖成了大脸仰天
    3. Dense: 亲好好养胃啊
    4. Hash: 我排便很顺畅我连着喝了一个月的红豆薏米汤感觉更通畅了
    5. ES: 我也是啊11每次想好吃几根的根本停不了啊啊啊啊啊啊啊啊啊啊啊啊
    
* Case 3:
    1. Context: 你妹的年终奖 [SEP] 哦还好还好有的单位据说还没有 [SEP] 你在缩我们公司嘛 [SEP] 妹纸我不是故意的
    2. Reference: 才不信就是故意的
    3. Dense: 我说我们公司不是你们我确定你看错了莫生气
    4. Hash: 唉说福利我就来气来公司半年了过年的年终奖知道发了多少么他妹的发了400还不够一顿请客钱呢
    5. ES: 你得寸进尺妹纸
    
* Case 4:
    1. Context: 有谁能提供下长沙复读学校的情况呢 [SEP] 长沙长郡明道高复还不错啊环境师资力量设施都很好的还有就是很安静食堂伙食比其他复读学校好多了 [SEP] 哈哈妹妹已经考上大学了一本谢谢大家了 [SEP] 亲你妹妹是在哪个学校复读的
    2. Reference: 大工希望与你离得不远
    3. Dense: 鲁东
    4. Dense: 鲁东
    5. ES: m直接延到2009年姐复读了啊亲
    
* Case 5:
    1. Context: 直播遇上了一个好平淡的男人姐不淡定了啊啊啊啊 [SEP] 大半夜咆哮啊亲 [SEP] 想到这时淡定不能啊姐是急性子啊 [SEP] 我要碎觉了亲晚安 [SEP] 嗯晚安亲我咆哮贴的第一个读者
    2. Reference: 加油加油啊坚强点洒脱些你可以滴
    3. Dense: 睡了亲扛不住了安
    4. Hash: 碎觉晚安亲
    5. ES: 乃见过卤煮真相么亲你忘了咆哮啊魂淡
    
* Case 6:
    1. Context: 阳澄湖大闸蟹是不是就是温州人讲的田丝儿啊 [SEP] 我觉得还是蝤蠓最好吃啊 [SEP] 是神马东东啊我都没听过 [SEP] 你肯定吃过
    2. Reference: 也许吧吃的时候不认识它
    3. Dense: 很久很久以前吃过一次我觉得美味只能品尝一次就再也没吃过了
    4. Hash: 没有印象记得吃过但是完全不记得了要么是时间太久要么是不够出色
    5. ES: 水产市场有不过你一定要阳澄湖的大闸蟹就不知道了
    
* Case 7:
    1. Context: 记录2012年桂林事业单位考试历程 [SEP] 好吧看了你的帖我顺手去报了个名 [SEP] 你报成功了吗 [SEP] 成功鸟啊填好资料上传了照片就等审核通过交钱啊 [SEP] 你报哪个职位呢
    2. Reference: 当然想报国税丫地点就不确定了你呢
    3. Dense: 上半年已经报过了下半年还没开始
    4. Hash: 油田公务员事业单位
    5. ES: 你报了吗就最后了亲
    
* Case 8:
    1. Context: 招聘深圳户外领队 [SEP] 你好今天刚看到你发的帖子想知道你们这工作是专职那还是兼职工作范围在哪里 [SEP] 本人目前自定为兼职吧 [SEP] 兼职为主但压力很大
    2. Reference: 自我化解下吧
    3. Dense: 不喜欢压力太大的地方比较自我希望有一定的自己支配的时间和空间
    4. Hash: 正在努力找兼职中主要是空闲时间总是跟工作时间不能吻合
    5. ES: 不好意思你再怎么说都一样的条件摆在那儿还有我没说我是兼职兼职的时间更没法跟这比了
    
* Case 9:
    1. Context: 千千公主4kraw格式福利挑选中 [SEP] _url_ [SEP] 看看你感觉真实的话给我传一份儿啊 [SEP] 唇唇你感觉真实就给我豆油来一份 [SEP] 给我也来一份
    2. Reference: 你真逗收豆油吧
    3. Dense: 来来来寄给我吧
    4. Hash: 来来来寄给我吧
    5. ES: 唇唇是谁
    
* Case 10:
    1. Context: 从浙江温州到甘肃兰州最佳交通路线是什么 [SEP] lz来吧来吧老乡在这里等你 [SEP] 好吧看在老乡的份上先住你那里嘿 [SEP] 好吧好吧你倒哪个学校了
    2. Reference: 文理你应该不知道
    3. Dense: 浙工大我会瞎说么
    4. Hash: 铜陵学院
    5. ES: 对呀老乡么看在老乡的份儿上围脖君你就给lz分享的音乐带个播放器吧

#### 3.2.3 Zh50w Dataset

* Case 1:
    1. Context: 健康太重要了，所以有时间尽量多锻炼身体 [SEP] 嗯，是的，我腿疼，也不能锻炼。 [SEP] 对了，你的腿伤还没恢复好，不急，慢慢调养！ [SEP] 也烦，快三个月了。 [SEP] 刚天，好好养，不要留下后遗症！
    2. Reference: 谢谢，谢谢关心！早安！
    3. Dense: 嗯嗯，抽时间还是去复查下吧，图个安心。
    4. Dense: 嗯，最近一直在坚持，高强度的间歇运动。
    5. ES: 嗯，顺的恢复快
    
* Case 2:
    1. Context: 哈哈哈哈哈哈哈哈哈你个瓜皮笑死我了哈哈哈 [SEP] 你这个智障猎人我气一年的 [SEP] 不带个人恩怨的来一局？
    2. Reference: 开好房叫我
    3. Dense: 你想多了好友王者一大把带飞的
    4. Hash: 不不不，王者选手也带不动我这个青铜玩家了
    5. ES: 哈哈哈哈哈可以！跟这种智障，你有什么可说的
    
* Case 3:
    1. Context: 请问第八张是什么？北方没见过 [SEP] 三角梅金边白。
    2. Reference: 十分感谢
    3. Dense: 谢谢啦！好美。
    4. Hash: 谢谢啦！好美。
    5. ES: 我知道，我是说三角梅在哪个地方
    
* Case 4:
    1. Context: 这么好 [SEP] 超级好吃
    2. Reference: 啊啊啊啊啊想吃想吃！隔着屏幕都闻到香味了
    3. Dense: 能吃到就心满意足了
    4. Hash: 哈哈我还没吃过有空出去逛街我也尝尝
    5. ES: 我也是超级好吃
    
* Case 5:
    1. Context: 早晨好红枫哥 [SEP] 百合上午好，谢谢你，开心快乐！ [SEP] 干嘛那么客气呢
    2. Reference: 不是客气，当大的也得尊重小的。
    3. Dense: 开玩笑，客气啥微博不常上，微信多聊聊呀.
    4. Hash: 哈哈！不是客气，是礼貌，早上好
    5. ES: 百合，你要干嘛
    
* Case 6:
    1. Context: 眼镜好看 [SEP] 人呢
    2. Reference: 和眼镜一样
    3. Dense: 你比眼镜好看！
    4. Hash: 你比眼镜好看！
    5. ES: 眼镜很文艺
    
* Case 7:
    1. Context: 图六是什么菜菜啊 [SEP] 豌豆尖
    2. Reference: 我们这边没有
    3. Dense: 全名土豆丝儿
    4. Hash: 算是韩国料理吧
    5. ES: 豌豆黄看着好好吃呀
    
* Case 8:
    1. Context: 只能说延安那两个长得最黑哈哈哈哈笑死我了 [SEP] 就你话多 [SEP] 自己承认了啊哈哈哈 [SEP] 你闭嘴啊 [SEP] 给我一个理由
    2. Reference: 眼睛小
    3. Dense: 不要给你的乱找这么。的理由行不
    4. Hash: 因为你就是这么可爱说不出理由
    5. ES: 我就喜欢黑你哈哈哈哈
    
* Case 9:
    1. Context: 谢明姣姐欣赏！这么晚还没睡哪？ [SEP] 每天都很迟睡晚上好
    2. Reference: 同好！你的微博很多都是凌晨发的。
    3. Dense: 原来如此！不行了姐老了姐得睡觉了
    4. Hash: 失眠好痛苦，姐。没想到你也那么晚、早知道就和你聊聊了
    5. ES: 姣姣早安，新天愉快！
    
* Case 10:
    1. Context: 求教朋友送我了青口干，亲知道该怎么做吗？ [SEP] 青口干，也叫做蚝豉，可以煲汤、煮粥、焖冬菇、煮粉丝粉条试试看哦
    2. Reference: 谢谢啦
    3. Dense: 好的，谢谢哦，我试试看
    4. Hash: 好的，谢谢分享
    5. ES: 煲汤还好，切薄片，最后放，煲十分钟

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
* coherence ratio 重新计算