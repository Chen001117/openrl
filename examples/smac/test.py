import numpy as np
import json

all_dict = dict()

with open("test.txt", "r") as input_file:
    input_file.readline()
    for i in range(1):
        with open("test2.txt", "w") as output_file:
            for _ in range(9):
                line = input_file.readline()
                if line[:5] == "     ":
                    line = line[5:]
                output_file.writelines(line[:-1])
        
        with open('test2.txt', "r") as input_file2: 
            data = input_file2.read() 
        out_dict = json.loads(data)
        
        # out_dict["ally_start_positions"]["item"] = np.array(out_dict["ally_start_positions"]["item"])
        # out_dict["enemy_start_positions"]["item"] = np.array(out_dict["enemy_start_positions"]["item"])
        
        all_dict[i] = out_dict
        
        for _ in range(4):
            input_file.readline()
    
with open("test.json", "w") as output_file:
    json.dump(all_dict, output_file)