from Template import *
from utils import *
from scipy.stats import pearsonr

class Residue(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residue, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y = Y + X
        return self.relu(Y)

def resgp_block(input_channels, num_channels, num_residuals, first_block = False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residue(input_channels, num_channels, use_1x1conv=True, strides=2)
            )
        else:
            blk.append(
                Residue(num_channels, num_channels)
            )
    return blk

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.matmul(queries, keys.transpose(-1, -2))/math.sqrt(d)
        self.attention_weight = torch.softmax(scores, dim=-1)
        f = torch.matmul(self.dropout(self.attention_weight), values)
        return f

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, d_k, d_v, h, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.input_dim = input_dim
        self.d_k = d_k
        self.d_v = d_v
        self.fc_q = nn.Linear(input_dim, h * d_k)
        self.fc_k = nn.Linear(input_dim, h * d_k)
        self.fc_v = nn.Linear(input_dim, h * d_v)
        self.fc_o = nn.Linear(h * d_v, input_dim)
        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(input_dim)
    
    def forward(self, queries, keys, values):
        batch_size, num_queries = queries.shape[:2]
        num_keys = keys.shape[1]

        q = self.fc_q(queries).view(batch_size, num_queries, self.h, self.d_k).permute(0, 2, 1, 3) #(batch_size, head, num, dimension)
        k = self.fc_k(keys).view(batch_size, num_keys, self.h, self.d_k).permute(0, 2, 1, 3)
        v = self.fc_v(values).view(batch_size, num_keys, self.h, self.d_v).permute(0, 2, 1, 3)
        attention = ScaledDotProductAttention(dropout=self.dropout)
        att = attention(q, k, v) # (batch_size, head, num, value)
        att = att.permute(0, 2, 1, 3).contiguous().view(batch_size, num_queries, self.h * self.d_v) #concat
        output = self.fc_o(att) # (batch_size, num, embedding)
        return self.layer_norm(queries + output)
    
class MLP(nn.Module):
    def __init__(self, length, hidden_dim):
        super().__init__()
        self.device = torch.device(cfg['cuda'] if torch.cuda.is_available() else 'cpu')
        self.mlp = nn.Sequential(
            nn.Linear(length, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 512)
        )
    def forward(self, x):
        output = self.mlp(x)
        return output

class Res_Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device(cfg['cuda'] if torch.cuda.is_available() else 'cpu')
        self.b1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=21, stride=11, padding=10),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5, padding=2)
        )
        self.b2 = nn.Sequential(*resgp_block(3, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resgp_block(64, 128, 2))
        self.b4 = nn.Sequential(*resgp_block(128, 256, 2))
        self.b5 = nn.Sequential(*resgp_block(256, 512, 2))
        self.b6 = nn.Sequential(*resgp_block(512, 512, 2))
        self.b7 = nn.Sequential(*resgp_block(512, 512, 2))
        self.resnet = nn.Sequential(
            self.b1, self.b2, self.b3, self.b4, self.b5,
            self.b6, self.b7
        )
        self.mulattn = MultiHeadAttention(512, 512, 512, 8)
    def forward(self, x):
        output = self.resnet(x).transpose(1, 2)
        output = self.mulattn(output, output, output)
        return output


class iADE(nn.Module):
    def __init__(self, length, hidden_dim):
        super().__init__()
        self.encoder = Res_Attn()
        self.decoder = MLP(length, hidden_dim)
        self.res_mlp = nn.Sequential(
            nn.Linear(length, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 512)
        )
        self.mulattn = MultiHeadAttention(512, 512, 512, 8)
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 64),
            nn.Dropout(0.3),
            nn.Linear(64, 1)       
        )
    
    def forward(self, x):
        snp = x['snp'].float()
        knowledged_snp = x['knowledged_snp'].float().transpose(1, 2)
        enc_output = self.encoder(knowledged_snp)
        dec_embedding = self.decoder(snp).unsqueeze(1)
        res = self.res_mlp(snp)
        dec_output = self.mulattn(dec_embedding, enc_output, enc_output).squeeze(1)
        dec_output = dec_output + res
        output = self.mlp(dec_output)
        return output

class iADEP(Template):
    def __init__(self, cfg, build = True):
        super().__init__(cfg)
        snp_length = cfg['length']
        self.model = iADE(snp_length, 1024)

        if build:
            self.build()
    
    def build(self):
        # Move model to GPU or CPU
        self.to(self.device)

        # Setup optimizers
        self.setup_optimizers(self.optimizer_name, self.optimizer_kwargs)
    
    # ----- rewrite Abstract class method ----- #
    def get_data_dict_from_dataloader(self, data):
        snp = data['snp'].to(self.device)
        knowledged_snp = data['knowledged_snp'].to(self.device)
        y = data['phen'].reshape(-1, 1).to(self.device)

        inputs = {'snp': snp,
                  'knowledged_snp': knowledged_snp}
        labels = {'phen': y}

        data_dict = {
            'inputs': inputs,
            'labels': labels
        }
        return data_dict
    
    def loss(self, outputs, data_dict):
        y = data_dict['labels']['phen']
        y_hat = outputs['y_hat']
        # loss
        loss = nn.MSELoss()(y, y_hat)
        # calculate alpha

        loss_details = {}
        return loss, loss_details
    
    def analyse_predictions(self, y_true, y_pred, info={}):
        y_true = y_true['phen'].squeeze()
        y_pred = y_pred['y_hat'].squeeze()
        assert y_true.shape == y_pred.shape, print('y_true: %s, y_pred: %s' % (y_true.shape, y_pred.shape))
        metrics_all = {}
        epoch_loss = info['epoch_loss']
        phase = info['phase']
        metrics_all['%s_epoch_loss' % phase] = epoch_loss
        mse = np.mean((y_pred - y_true) ** 2)
        mae = np.mean(np.abs(y_pred - y_true))
        pcc = pearsonr(y_pred, y_true)[0]
        metrics_all['mse'] = mse
        metrics_all['pcc'] = pcc
        metrics_all['mae'] = mae
        return metrics_all
    
    # ----- forward ----- #
    def forward(self, x):
        outputs = {}
        outputs['y_hat'] = self.model(x)
        return outputs