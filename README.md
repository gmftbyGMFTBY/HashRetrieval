# HashRetrieval
Learning to Hash for Coarse Retrieval in Open-Domain Dialog Systems

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
### 2.1 Comparsion between Term-Frequency and dense vector retrieval
Compare the performance and the time cost

### 2.2 Comparsion between the dense vector and hash vector retrieval
Compare the performance and the time cost

### 2.3 Overall comparsion
cross-bert post rank with different coarse retrieval strategies:
* Term-Frequency
* Dense vector
* Hash vector