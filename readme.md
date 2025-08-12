## Project Name

This is the code for FairSplit algorithm. 


### Installation

pip install -r requirements.txt


### How to Run

**Example**

```bash
python main.py --dataset = Recidivism --model=gcn --epochs=2000 --max_nodes=10000 --init_lr=0.001 --weight_decay=1e-05 --dropout=0.5 --data_folder=/path/to/folder/```

```


Change the hyperparameter values, and most importantly the data folder path

List of allowed values for arguments:

&nbsp; &nbsp; &nbsp; &nbsp; dataset :  Recidivism, Credit, German

&nbsp; &nbsp; &nbsp; &nbsp; model   :  gcn, sage, appnp


### License

MIT
	
