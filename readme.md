## Project Name
&nbsp; 
This is the code for FairSplit algorithm. 
&nbsp; 

### Installation
&nbsp; 
pip install -r requirements.txt
&nbsp; 

### How to Run
&nbsp; 
**Example**
*python main.py --dataset=Recidivism --model=gcn --epochs=2000 --max_nodes=10000 --init_lr=0.001 --weight_decay=1e-05 --dropout=0.5 --data_folder=/path/to/folder/*
&nbsp; 
Change the hyperparameter values, and most importantly the data folder path
List of allowed values for arguments:
&nbsp; &nbsp; &nbsp; &nbsp; dataset :  Recidivism, Credit, German
&nbsp; &nbsp; &nbsp; &nbsp; model   :  gcn, sage, appnp
&nbsp; 

### License
MIT
	
