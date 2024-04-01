import torch
import numpy as np
from torch.utils.data import Dataset

def weight_initialization(snp_data, gwas_value, lit_value):
    # check
    assert snp_data.shape[1] == len(gwas_value) == len(lit_value)
    lit_pvalue = torch.as_tensor(-np.log10(lit_value)).reshape(-1, 1)
    gwas_pvalue = torch.as_tensor(-np.log10(gwas_value)).reshape(-1, 1)
    ones_value = torch.ones(snp_data.shape).unsqueeze(2)
    snp_train_ad = (snp_data - 1).unsqueeze(2)
    snp_train_do = abs(abs(snp_train_ad)-1)
    snp_data = torch.cat([snp_train_ad, snp_train_do, ones_value], dim=2)
    domi_value = torch.ones([snp_data.shape[1], 1])
    weight_init = torch.cat([gwas_pvalue, domi_value, lit_pvalue], dim=1)
    snp_data = snp_data * weight_init
    return snp_data

class MilletDataset(Dataset):
    def __init__(self, snp, gwas, lit, phen):
        super().__init__()
        self.snp = snp
        self.phen = phen
        self.knowledged_snp = weight_initialization(snp, gwas, lit)
        self.data = {}
        self.data['snp'] = self.snp
        self.data['phen'] = self.phen
        self.data['knowledged_snp'] = self.knowledged_snp
    
    def __len__(self):
        return self.snp.shape[0]

    def __getitem__(self, index):
        items = {}
        for key, value in self.data.items():
            items.update({
                key: value[index]
            })
        return items
    
def extend_dicts(dict1, dict2):
    if len(dict1) == 0:
        for key, value in dict2.items():
            dict1[key] = value.data.cpu().numpy()
        return dict1
    
    assert set(dict1.keys()) == set(dict2.keys())
    for key, value in dict2.items():
        dict1[key] = np.concatenate([dict1[key], value.data.cpu().numpy()])
    return dict1