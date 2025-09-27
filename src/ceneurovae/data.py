import torch
from torch.utils.data import Dataset, DataLoader
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

class DatasetWindow(Dataset):
  def __init__(self, datasets, window_T: int, stride: int, ignore_unlabeled: bool = True):
    self.datasets = datasets
    self.window_T = int(window_T)
    self.stride   = int(stride)
    self.ignore_unlabeled = bool(ignore_unlabeled)

    # determine which neuron rows to keep per dataset (labeled only if requested)
    self.keep_idx = []  # list[Tensor(long) of indices per dataset]
    for dataset in self.datasets:
      I = torch.as_tensor(dataset["I"]).long()
      if self.ignore_unlabeled:
        L = torch.nonzero(I > 0, as_tuple=True)[0]
      else:
        L = torch.arange(I.numel(), dtype=torch.long)
      self.keep_idx.append(L)

    # build (dataset_idx, t_start) only for datasets that have something to train on
    self.index: List[Tuple[int, int]] = []
    for i_dataset, dataset in enumerate(self.datasets):
      # if ignoring unlabeled and none are labeled, skip this dataset entirely
      if self.ignore_unlabeled and self.keep_idx[i_dataset].numel() == 0:
        continue
      T = int(dataset["X"].shape[1])
      for t_start in range(0, T - self.window_T + 1, self.stride):
        self.index.append((i_dataset, t_start))

  def __len__(self):
    return len(self.index)

  def __getitem__(self, k: int):
    i_dataset, t_start = self.index[k]
    dataset = self.datasets[i_dataset]
    t_end = t_start + self.window_T

    # slice time first
    X = torch.as_tensor(dataset['X'][:, t_start:t_end])
    M = torch.as_tensor(dataset['M'][:, t_start:t_end])
    Bx = torch.as_tensor(dataset['Bx'][:, t_start:t_end])
    I = torch.as_tensor(dataset['I']).long()

    # slice neurons (drop unlabeled rows if requested)
    L = self.keep_idx[i_dataset]
    X = X[L] # (N_kept, T)
    M = M[L] # (N_kept, T)
    I = I[L] # (N_kept,)

    metadata = {'uid': dataset.get('uid', dataset), 't_start': t_start}

    return X, M, Bx, I, metadata
  
def pad_collate(batch):
    Xs, Ms, Bxs, Is, metas = zip(*batch)
    B = len(batch)
    Ns = [x.shape[0] for x in Xs]
    T  = Xs[0].shape[1]
    Tb = Bxs[0].shape[0] if Bxs[0].numel() > 0 else 0
    Nmax = max(Ns)
    dev = Xs[0].device

    X_pad  = torch.zeros(B, Nmax, T, device=dev)
    M_pad  = torch.zeros(B, Nmax, T, device=dev)
    I_pad  = torch.zeros(B, Nmax, dtype=torch.long, device=dev)
    Bx_pad = torch.zeros(B, Tb, T, device=dev) if Tb > 0 else torch.empty(B, 0, T, device=dev)

    for b in range(B):
        n = Ns[b]
        X_pad[b, :n] = Xs[b]
        M_pad[b, :n] = Ms[b]
        I_pad[b, :n] = Is[b].long() # ensure correct dtype for embedding
        if Tb > 0:
            Bx_pad[b] = Bxs[b]

    return X_pad, M_pad, Bx_pad, I_pad, metas

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

def build_loaders(datasets, window_T=200, stride=50, batch_size=8, num_workers=0,
                  val_frac: float = 0.2, seed: int = 0, ignore_unlabeled: bool = True):
  train, val, train_uids, val_uids = split_datasets(datasets, val_frac=val_frac, seed=seed)

  dataset_train = DatasetWindow(train, window_T=window_T, stride=stride,
                                ignore_unlabeled=ignore_unlabeled)
  dataset_val = DatasetWindow(val,   window_T=window_T, stride=stride,
                                ignore_unlabeled=ignore_unlabeled)

  loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                            collate_fn=pad_collate, num_workers=num_workers, pin_memory=True)
  loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                            collate_fn=pad_collate, num_workers=num_workers, pin_memory=True)

  return loader_train, loader_val, train_uids, val_uids