# create .bat file for shell

import os

file_path = "D:/Indranil/JRF/Submission/FairSplit/codebase/FairSplit/run.bat"
if os.path.exists(file_path):  # Check if file exists
    os.remove(file_path)       # Delete the file

epochs = 2000 
command_count = 0
with open(file_path, "a") as file:
    for dataset in ['Recidivism', 'Credit', 'German']:
        max_nodes = {'German': 1000, 'Recidivism': 10000, 'Credit': 15000}.get(dataset)
        for model in ['gcn', 'sage', 'appnp']: # sage to be added later            
            for init_lr in [0.01, 0.001, 0.0001]:
                for weight_decay in [0.001, 0.0001, 0.00001]:
                    for dropout in [0.2, 0.5]:
                        command_str = f"python main.py --dataset={dataset} --model={model}"
                        command_str = f"{command_str} --epochs={epochs} --max_nodes={max_nodes}"
                        command_str = f"{command_str} --init_lr={init_lr} --weight_decay={weight_decay}" 
                        command_str = f"{command_str} --dropout={dropout}"                                        
                        file.write(command_str + "\n")  
                        command_count += 1
print("\nTotal commands generated: ", command_count)
