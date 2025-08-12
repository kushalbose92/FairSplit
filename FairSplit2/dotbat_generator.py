# create .bat file for shell

import os

file_path = 'D:/Indranil/ML2/projects/fairness/FairSplit/FairSplit2/run_all.sh'
if os.path.exists(file_path):  # Check if file exists
    os.remove(file_path)       # Delete the file

epochs = 2000 
command_count = 0
model_hetero = 'linkx'
with open(file_path, "a") as file:
    for dataset in ['Credit']:
        for model_homo in ['gcn', 'appnp', 'sage']:
            max_nodes = {'German': 1000, 'Recidivism': 10000, 'Credit': 15000}.get(dataset)
            for init_lr in [0.01, 0.001, 0.0001]:
                for weight_decay in [0.001, 0.0001, 0.00001]:
                    for dropout in [0.2, 0.5]:
                        command_str = f"python main.py --dataset={dataset} --model_homo={model_homo}"
                        command_str = f"{command_str} --model_hetero={model_hetero} --epochs={epochs}"
                        command_str = f"{command_str} --max_nodes={max_nodes} --init_lr={init_lr}" 
                        command_str = f"{command_str} --weight_decay={weight_decay} --dropout={dropout}"                                        
                        file.write(command_str + "\n")  
                        command_count += 1
print("\nTotal commands generated: ", command_count)
