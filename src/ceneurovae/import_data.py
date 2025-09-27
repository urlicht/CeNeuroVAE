import h5py
import torch
import numpy as np

def convert_data_to_tensor(datasets):
    for k, v in datasets.items():
        v["X"]  = torch.tensor(v["X"], dtype=torch.float32)
        v["M"]  = torch.tensor(v["M"], dtype=torch.float32)
        v["Bx"] = torch.tensor(v["Bx"], dtype=torch.float32)
        v["I"]  = torch.tensor(v["I"], dtype=torch.long)
    
    return datasets

def import_h5(path):
    data = {}
    with h5py.File(path, "r") as f:
        g_data = f["data"]
        B_norm_divisor = f["B_norm_divisor"][()].reshape((-1,1))
        
        for uid_key in g_data.keys():
            dataset = g_data[uid_key]
            inner_dict = {}
            for item_key in dataset.keys():
              inner_dict[item_key] = dataset[item_key][()]
            inner_dict["uid"] = uid_key
            if "M" not in inner_dict or inner_dict["M"] is None:
              inner_dict["M"] = np.ones_like(inner_dict["X"])

            inner_dict["Bx"] = inner_dict["B"] / B_norm_divisor
            data[uid_key] = inner_dict
        labels = f["label_name"][()].astype(str).tolist()  # list of str

    return convert_data_to_tensor(data), labels