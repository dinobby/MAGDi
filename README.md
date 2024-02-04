# MAGDI: Structured Distillation of Multi-Agent Interaction Graphs Improves Reasoning in Smaller Language Models
![Image](https://i.imgur.com/alEYjUs.png)

## Installation
This project is built on Python 3.10.11. All dependencies can be installed via:

`pip install -r requirements.txt`

## Project Directory Structure
To run this project, the complete directory structure would be
```
MAGDi/
├── checkpoints
    ├── MAGDi_ARC/
    ├── MAGDi_ECQA/
    ├── MAGDi_GSM8K/
    ├── MAGDi_MATH/
    └── MAGDi_SQA
├── MAG/
    ├── ARC_1000.json
    ├── ECQA_1000.json
    ├── GSM8K_1000.json
    ├── MATH_1000.json
    └── SQA_1000.json
├── node_emb/
    ├── ARC_node_emb.pkl
    ├── ECQA_node_emb.pkl
    ├── GSM8K_node_emb.pkl
    ├── MATH_node_emb.pkl
    └── SQA_node_emb.pkl
├── test_data/
    ├── ARC_test.json
    ├── ECQA_test.json
    ├── GSM8K_test.json
    ├── MATH_test.json
    └── SQA_test.json
├── data_utils.py
├── get_node_emb.py
├── model.py
├── test.py
├── train.py
└── utils.py
```

For `checkpoints`, `MAG`, `node_emb` and `test_data`, you can download them via this link: [Google Drive](https://drive.google.com/drive/folders/187_s_I0e3NUJGFeV0AHzG85PrTotSTuF?usp=sharing)

## Running MAGDi

### Step 1: Prepare MAGs Training Data

![Image](https://i.imgur.com/Ll3AmNL.png)

We provide 1000 samples for each dataset (StrategyQA, CommonsenseQA, ARC-Challenge, GSM8K, MATH)

These samples are in MAG format. You can download them via this link: [Google Drive](https://drive.google.com/drive/folders/187_s_I0e3NUJGFeV0AHzG85PrTotSTuF?usp=sharing)

### Step 2: Get Node Embeddings
Node embeddings are initialized by an average pooling over the reasoning sequence.

Run `get_node_emb.py` to obtain the initial node embedding.

Or download the node embeddings via this link: [Google Drive](https://drive.google.com/drive/folders/187_s_I0e3NUJGFeV0AHzG85PrTotSTuF?usp=sharing)

### Step 3: Train the Base Student Model using MAGDi
Run ```train.py --dataset SQA --num_epochs 10 --lr 5e-6```

For more configuration and hyperparameters, please refer to `train.py`

### Step 4: Evaluate the MAGDi-Augmented Model
Run ```test.py --dataset SQA --batch_size 256 --temperature 0.7 --max_new_tokens 300```

You can find the trained checkpoints and test data here: [Google Drive](https://drive.google.com/drive/folders/187_s_I0e3NUJGFeV0AHzG85PrTotSTuF?usp=sharing)

For MATH dataset, you may need to set a lower `batch_size` and larger `max_new_tokens`, 

e.g., `batch_size` = 10 and `max_new_tokens` = 700.

## Citation
```
@article{chen2023magdi,
  title={MAGDi: Structured Distillation of Multi-Agent Interaction Graphs Improves Reasoning in Smaller Language Models},
  author={Chen, Justin Chih-Yao and Saha, Swarnadeep and Stengel-Eskin, Elias and Bansal, Mohit},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```
