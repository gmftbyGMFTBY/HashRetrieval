The data are collected from GPT2-chitchat: https://github.com/yangjianxin1/GPT2-chitchat.git

The raw zh50w dataset are not used for response selection task, run the following commands to transform the dataset format:

1. Write the data into the elasticsearch
```bash
python transform_retrieval.py --mode train
python transform_retrieval.py --mode test
```
