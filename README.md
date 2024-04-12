# CS5260Assignment6 Resnet Training on Cifar10 with ColossalAI

### The model used in the experiment: 
Resnet18
### The dataset employed: 
Cifar10
### How to run: 
pip install -r requirement.txt

colossalai run --nproc_per_node 1 mytrain.py
### Experiment results:

Results after 10 epochs of training with 1 V100 GPU are shown below:
Loss is calculated via CrossEntropyLoss function.

| Technique | Accuracy | Loss | Time | Memory |
|----------|----------|----------|----------|----------|
| ColossalAI | 74.01% | 0.44 | 221.38s | 749.53 MB |
| Benchmark | 70.21% | 0.592 | 237.96s | 873.48 MB |

### Github repository link:
https://github.com/JosephMcjhin/CS5260Assignment6
