from iADEP import *
from utils import *
from Template import *
import argparse
from torch.utils.data import DataLoader

# argparse and cfg
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--trait', type=str, default='TSLL')
parser.add_argument('--checkpoint', type=str, default='./checkpoint/TSLL.pt')
args = parser.parse_args([])
cfg = {
    'trait': args.trait,
    'cuda': args.cuda,
    'verbose':{
        'layer_magnitudes': False,
        'metrics': True
    },
    'batch': args.batch,
    'epochs': args.epochs,
    'seed': args.seed,
    'optimizer_name': 'adam',
    'optimizer_kwargs': {
        'lr': args.lr,
        'weight_decay': 1e-6
    },
    'model_checkpoint_path': args.checkpoint
}
# set seed
torch.manual_seed(cfg['seed'])
torch.cuda.manual_seed(cfg['seed'])
trait = cfg['trait']

# load data
snp_train = pd.read_csv('../example_data/millet827/kfold/fold1/snp_train.csv', index_col=0).values
snp_test = pd.read_csv('../example_data/millet827/kfold/fold1/snp_test.csv', index_col=0).values
phen_train = pd.read_csv('../example_data/millet827/kfold/fold1/phen_train.csv', index_col=0)[trait].values
phen_test = pd.read_csv('../example_data/millet827/kfold/fold1/phen_test.csv', index_col=0)[trait].values
lit = pd.read_csv('../example_data/millet827/kfold/fold1/lit_result.csv', index_col=0)[trait + '_lit'].values
gwas = pd.read_csv('../example_data/millet827/kfold/fold1/gwas_result.csv', index_col=0)[trait].values
cfg['length'] = snp_train.shape[1]
# convert to tensor
snp_train = torch.as_tensor(snp_train, dtype=torch.long)
snp_test = torch.as_tensor(snp_test, dtype=torch.long)
phen_train = torch.as_tensor(phen_train, dtype=torch.float32)
phen_test = torch.as_tensor(phen_test, dtype=torch.float32)
phen_mean = phen_train.mean()
phen_train = phen_train - phen_mean
phen_test = phen_test - phen_mean
snp_train = MilletDataset(snp_train, gwas, lit, phen_train)
snp_test = MilletDataset(snp_test, gwas, lit, phen_test)
train_dataloader = DataLoader(snp_train, batch_size=cfg['batch'])
test_dataloader = DataLoader(snp_test, batch_size=cfg['batch'])

# train and test
dataloaders = {
    'train': train_dataloader,
    'test': test_dataloader
    }
datasizes = {
    'train': len(snp_train),
    'test': len(snp_test)
}
model = iADEP(cfg)
output = model.fit(dataloaders, datasizes)