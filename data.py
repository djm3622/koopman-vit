import torch
import os
import numpy as np
import scipy as sp


def main(path, device_pref, solver, fixed_seq_len, ahead, tail, device_ind):
    instances = get_num_instances(path)
    seq_len_cache = get_training_lens(path, instances, solver, fixed_seq_len)
    file_cache = get_training_paths(path, instances, seq_len_cache, solver)
    
    print_diagnostics(seq_len_cache, file_cache)
    
    device = set_device(device_pref, device_ind)
    
    train_set, valid_set = train_test_split(seq_len_cache)
    (x_train_data, y_train_data) = setup_pairs(train_set, file_cache, ahead, tail)
    (x_valid_data, y_valid_data) = setup_pairs(valid_set, file_cache, ahead, tail)
    
    print(f'Train size: {x_train_data.shape[0]}, Percent of toal: '
      f'{(x_train_data.shape[0] / (x_train_data.shape[0] + x_valid_data.shape[0])) * 100:.2f}%, '
      f'Unique instances: {len(train_set)}')
    print(f'Train size: {x_valid_data.shape[0]}, Percent of toal: '
      f'{(x_valid_data.shape[0] / (x_train_data.shape[0] + x_valid_data.shape[0])) * 100:.2f}%, '
      f'Unique instances: {len(valid_set)}')
    
    return device, (x_train_data, y_train_data), (x_valid_data, y_valid_data)


def set_device(device_pref, device_ind):
    device = None
    
    if device_pref == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_ind}')
        print('Now using GPU.')
    else:
        device = torch.device('cpu')
        if device_pref == 'cuda':
            print('GPU not available, defaulting to CPU.')
        else:
            print('Now using CPU.')
    
    return device


def get_num_instances(folder):
    inst = set()
    
    for filename in os.listdir(folder):
        if filename.split('_')[2] in inst:
            continue
        inst.add(int(filename.split('_')[2]))
    
    return inst


def get_training_lens(folder, instances, solver, fixed_seq_len):
    inst_len_cache = {}
    
    for instance in instances:
        x = os.path.join(folder, f'{solver}_x_{instance}_')
        
        for seq in range(0, fixed_seq_len+1):
            x_seq = x + f'{seq}.npy'
            
            if not os.path.isfile(x_seq):
                # If finished early use timesteps 10 before.
                if seq != fixed_seq_len:
                    inst_len_cache[instance] = seq-10
                else:
                    inst_len_cache[instance] = seq-1
                break
                
    return inst_len_cache


def get_training_paths(folder, instances, inst_len_cache, solver):
    file_cache = {}
        
    for instance in instances:
        for seq in range(inst_len_cache[1]+1):
            x_file = f'x_{instance}_{seq}'
            
            file_cache[x_file] = os.path.join(folder, f'{solver}_{x_file}.npy')
    
    return file_cache


def print_diagnostics(seq_len_cache, file_cache):
    print(f'Number of file paths cached: {len(file_cache)}')
    print('Key: x_{i}_{n}, Value: /data/users/jupyter-dam724/datadump(colliding)/ros2_x_{i}_{m}.npy\n')

    print('Inst | Seq. Len. \n================')
    for key, val in seq_len_cache.items():
        if val < 215:
            # Denote the true value of problem instances. Investigate if need be.
            print(f'{key:4} | {val:3} <-- {val+10}')
        else:
            print(f'{key:4} | {val:3}')
            
            
def setup_pairs(seq_len_cache, file_cache, ahead, tail):
    x, y = [], []
    
    for instance, sequences in seq_len_cache.items():
        for seq in range(sequences+1):
            if seq > sequences-(ahead+tail):
                break
            
            inpts = []
            for inpt in range(ahead):
                inpts.append(file_cache[f'x_{instance}_{seq+inpt}'])
            
            targs = []
            for targ in range(tail):
                targs.append(file_cache[f'x_{instance}_{seq+ahead+targ}'])
                
            x.append(inpts)
            y.append(targs)
            
    return np.array(x), np.array(y)


def train_test_split(seq_len_cache):
    train, test = {}, {}
    for key in seq_len_cache:
        if key < 241:
            test[key] = seq_len_cache[key]
        else:
            train[key] = seq_len_cache[key]
        
    return train, test

