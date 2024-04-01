import time
import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from utils import *
from torch import optim
from torch.autograd import Variable

class Template(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device( cfg['cuda'] if torch.cuda.is_available() else 'cpu')
        self.verbose = cfg['verbose']
        self.epochs = cfg['epochs']
        self.seed = cfg['seed']
        self.optimizer_name = cfg['optimizer_name']
        self.optimizer_kwargs = cfg['optimizer_kwargs']
        self.checkpoint = cfg['model_checkpoint_path']
        self.batch = cfg['batch']

    # ----- Abstract class methods ----- #
    def forward(self, X):
        raise NotImplementedError()

    def get_data_dict_from_dataloader(self, data, phase):
        raise NotImplementedError()
    
    def loss(self, output, data_dict):
        raise NotImplementedError()
    
    def analyse_predictions(self, y_true, y_pred, info={}):
        raise NotImplementedError()
    
    # ----- Standard deep learning step ----- #
    def train_or_eval_dataset(self, dataloaders, dataset_sizes, phase):
        assert phase in ['train', 'test'], print('Wrong phase!')
        if phase == 'train':
            self.train(True)
        else:
            self.train(False)
        
        running_loss = 0.0
        n_batches_loaded = 0
        loss_details = []
        concatenated_labels = {}
        concatenated_outputs = {}

        # get the inputs
        for data in dataloaders[phase]:
            n_batches_loaded += 1
            data_dict = self.get_data_dict_from_dataloader(data)
            inputs = data_dict['inputs']
            labels = data_dict['labels']
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward
            outputs = self.forward(inputs)

            # compute loss
            loss, loss_detail = self.loss(outputs, data_dict)
            loss_details.append(loss_detail)

            # record labels and outputs
            concatenated_labels = extend_dicts(concatenated_labels, labels)
            concatenated_outputs = extend_dicts(concatenated_outputs, outputs)

            # backward and optimize
            if phase == 'train':
                loss.backward()
                self.optimizer.step()
            
            # loss statistics
            running_loss += loss.data.item() * labels[list(labels.keys())[0]].size(0) # Mean batch loss -> batches loss
        
        epoch_loss = running_loss / dataset_sizes[phase] # sum batches loss -> Mean epoch loss
        info = {
            'phase': phase,
            'dataset_size': dataset_sizes[phase],
            'epoch_loss': epoch_loss,
            'loss_details': loss_details
        }
        metrics_for_epoch = self.analyse_predictions(concatenated_labels, concatenated_outputs, info)
        if self.verbose['metrics']:
            print(metrics_for_epoch)
        
        return metrics_for_epoch

    
    def fit(self, dataloaders, dataset_sizes):
        since = time.time()
        all_metrics = {}


        for epoch in range(self.epochs):
            epoch_t0 = time.time()
            print('\nEpoch {}/{}'.format(epoch, self.epochs - 1))
            print('-' * 60)
            metrics_for_epoch = {}

            # train one epoch
            metrics_for_phase = self.train_or_eval_dataset(dataloaders, dataset_sizes, 'train')
            metrics_for_epoch.update(metrics_for_phase)


            all_metrics[epoch] = metrics_for_epoch
            print('Total second taken for epoch: %2.3fs' % (time.time() - epoch_t0))
        
            if self.verbose['layer_magnitudes']:
                print('\n\n Printing layer magnitudes')
                self.print_layer_magnitudes(epoch)

        torch.save(self.state_dict(), self.checkpoint)    
        all_metrics['final_results'] = metrics_for_epoch
        time_elapse = time.time() - since
        all_metrics['total_seconds_to_train'] = time_elapse
        print("Training complete in {:.0f}m, {:.0f}s".format(time_elapse // 60, time_elapse % 60))

        self.load_state_dict(torch.load(self.checkpoint))
        self.train(False)
        print('The test set metrics: ')
        test_metrics = self.train_or_eval_dataset(dataloaders, dataset_sizes, 'test')
        return test_metrics

        
    def setup_optimizers(self, optimizer_name, optimizer_kwargs):
        if optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()), **optimizer_kwargs
            )
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), **optimizer_kwargs
            )
        else:
            raise Exception('Not a valid optimizer')
        

    def print_layer_magnitudes(self, epoch):
        '''
        check whether each layer's L2 norm is updating
        '''
        for name, param in self.named_parameters():
            magnitude = np.linalg.norm(param.data.cpu()) # each layer's L2 norm
            if name not in self.layer_magnitudes:
                self.layer_magnitudes[name] = magnitude
                print('The magnitude of layer %s at epoch %i is %2.5f' % (name, epoch, magnitude))
            else:
                old_magnitude = self.layer_magnitudes[name]
                delta_magnitude = magnitude - old_magnitude
                print(
                    'The magnitude of layer %s at epoch %i is %2.5f (delta %2.5f from last epoch)'%(
                    name, epoch, magnitude, delta_magnitude
                    )
                )
                self.layer_magnitudes[name] = magnitude