import h5py
import random
import numpy as np
from typing import Dict, List

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

    return data

def split_datasets(datasets: List[Dict], val_frac: float = 0.2, seed: int = 0):
    uids = datasets.keys()
    uniq = sorted(set(uids))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_uids = set(uniq[:n_val])
    train_uids = set(uniq[n_val:])

    train = []
    val = []

    for (uid, dataset) in datasets.items():
      if uid in val_uids:
        val.append(dataset)
      else:
        train.append(dataset)
        
    return train, val, train_uids, val_uids