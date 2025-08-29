## FairSplit

This is the code for the FairSplit algorithm that partitions a graph based on homophilic and heterophilic edges and learns two different node representations . 
FairSplit is a framework for studying fairness in Graph Neural Networks (GNNs) using graph augmentation based on edge-splitting technique.  
It supports multiple downstream tasks such as unsupervised homophily estimation, link prediction and product recommendation.

---
   
## 📂 Project Structure
```
FairSplit/
│── FairSplit			# Unsupervised Homophily Estimation, Link Prediction 
│── FairSplit2			# Product Recommendation
│── requirements.txt	# Dependencies
│── README.md 			# Project info
```

---

## ⚙️ Installation

1. **Clone repository**
```bash
git clone https://github.com/kushalbose92/FairSplit/tree/main/FairSplit
cd Fairsplit
```

2. **Install dependencies**

> ⚠️ Important: PyTorch Geometric has special installation instructions.  
> First install PyTorch matching your CUDA version:  
> https://pytorch.org/get-started/locally/

Example (CUDA 11.8):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install PyG:
```bash
pip install torch-geometric torch-scatter torch-sparse
```

Finally:
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run experiments with:
```bash
python main.py --dataset = Recidivism --model=gcn --epochs=2000 --max_nodes=10000 --init_lr=0.001 --weight_decay=1e-05 --dropout=0.5 --data_folder=/path/to/folder/```
```

Results are saved in:
- `results/results.csv` → numerical metrics  
- `results/hist/` → training plots  

---

## 📊 Features
- Multiple datasets: German, Credit, Recidivism
- Models: GCN, GraphSAGE, APPNP
- Saves results in CSV for later analysis

---

## 👨‍💻 Contributors
- Indranil Ojha

