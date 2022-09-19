import math
import time
import random
import os
import argparse
import shutil
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from aiplatform.s3utility import S3Callback, S3Utils
from aiplatform.config import cfg as aip_cfg
from .model import Seq2Seq
from . import transforms
from .config import cfg
from .dataset import FlightDataset


# TODO: check tensor types


def calc_accuracy(output, Y, mask):
    """
    Calculate the accuracy (point by point evaluation)
    :param output: output from the model (tensor)
    :param Y: ground truth given by dataset (tensor)
    :param mask: used to mask out the padding (tensor)
    :return: accuracy used for validation logs (float)
    """
    _, max_indices = torch.max(output.data, 1)
    max_indices = max_indices.view(mask.shape[1], mask.shape[0]).permute(1, 0)
    Y = Y.view(mask.shape[1], mask.shape[0]).permute(1, 0)
    max_indices = torch.masked_select(max_indices, mask)
    Y = torch.masked_select(Y, mask)
    train_acc = (max_indices == Y).sum().item()/max_indices.size()[0]
    return train_acc, max_indices, Y

# def loss_function(trg, output, mask):
#     """
#     Calculate the loss (point by point evaluation)
#     :param trg: ground truth given by dataset (tensor)
#     :param output: output from the model (tensor)
#     :param mask: used to mask out the padding (tensor)
#     :return: loss needed for backpropagation and logging (float)
#     """
#     trg = trg[1:].permute(1,0,2)
#     output = output[1:].permute(1,0,2)
#     mask = mask.unsqueeze(2).expand(trg.size())
#     trg = torch.masked_select(trg, mask)
#     output = torch.masked_select(output, mask)
#     label_mask = (trg != 0)
#     selected = torch.masked_select(output, label_mask)
#     loss = -torch.sum(selected) / selected.size()[0]
#     return loss


def default_collate(batch, y_padding_value, mode3_padding_value, callsign_padding_value):
    """
    Stack the tensors from dataloader and pad sequences in batch
    :param batch: batch from the torch dataloader
    :return: stacked input to the seq2seq model
    """
    batch.sort(key=lambda x: x[-1], reverse=True)
    batch_x, batch_y, batch_mode3, batch_callsign, batch_len = zip(*batch)
    batch_pad_x = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True)
    batch_pad_y = torch.nn.utils.rnn.pad_sequence(
        batch_y, batch_first=True, padding_value=y_padding_value)
    batch_pad_mode3 = torch.nn.utils.rnn.pad_sequence(
        batch_mode3, batch_first=True, padding_value=mode3_padding_value)
    batch_pad_callsign = torch.nn.utils.rnn.pad_sequence(
        batch_callsign, batch_first=True, padding_value=callsign_padding_value)

    batch_len = torch.Tensor(batch_len).type(torch.int64).unsqueeze(1)
    return [batch_pad_x, batch_pad_y, batch_pad_mode3, batch_pad_callsign, batch_len]


class Experiment(object):

   # should init as arguments here
    def __init__(self, args, clearml_task=None):

        self.clearml_task = clearml_task
        self.datapath = args.data_path
        self.features = cfg['data']['features']
        self.callsign_column = args.data_identifiers_callsign_data_column
        self.mode3_column = args.data_identifiers_mode3_data_column
        self.time_encoding_dims = args.data_time_encoding_dims
        self.n_features = len(cfg['data']['features']) + \
            self.time_encoding_dims + 8
        self.label = args.data_label

        self.weight_by = args.data_weight_by

        self.d_model = args.model_d_model
        self.dim_feedforward = args.model_dim_feedforward
        self.nhead = args.model_nhead
        self.num_encoder_layers = args.model_num_encoder_layers
        self.num_decoder_layers = args.model_num_decoder_layers
        self.enc_dropout = args.model_enc_dropout
        self.dec_dropout = args.model_dec_dropout
        self.input_dropout = args.model_input_dropout
        self.transformer_activation = args.model_transformer_activation

        self.checkpoint_dir = args.train_checkpoint_dir
        self.batch_size = args.train_batch_size
        self.learning_rate = args.train_lr

        self.n_epochs = args.train_epochs
        self.auto_lr = args.train_auto_lr
        self.n_gpu = args.train_n_gpu
        self.accelerator = args.train_accelerator
        self.model_save_period = args.train_model_save_period
        self.log_every_n_steps = args.train_log_every_n_steps
        self.save_top_k = args.train_save_top_k
        self.num_workers = args.train_num_workers

        self.id_embed_dim = args.model_id_embed_dim
        self.n_mode3_token_embedding = args.model_n_mode3_token_embedding
        self.n_mode3_token_layers = args.model_n_mode3_token_layers

        self.n_callsign_token_embedding = args.model_n_callsign_token_embedding
        self.n_callsign_token_layers = args.model_n_callsign_token_layers

        self.seed = args.train_seed
        self.transforms = cfg['data']['transforms']
        self.lr_schedule = cfg['train']['lr_schedule']

    def _get_logger(self):
        logger = TensorBoardLogger(self.checkpoint_dir, name='logs')
        return logger

    def _get_callbacks(self):
        callbacks = []

        # checkpoint_callback = CustomCheckpoint(
        #     task_name=self.clearml_task.name,
        #     dirpath=self.checkpoint_dir,
        #     filename = '-{epoch}',
        #     save_top_k= self.save_top_k,
        #     verbose=True,
        #     monitor='val_loss',
        #     mode='min',
        #     period = self.model_save_period
        #     )
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename='{k}-{epoch}',
            save_top_k=self.save_top_k,
            verbose=True,
            save_last=True,
            monitor='val_loss',
            mode='min',
            every_n_val_epochs=self.model_save_period
        )
        callbacks.append(checkpoint_callback)
        if self.lr_schedule['scheduler']:
            lr_logging_callback = LearningRateMonitor(logging_interval='step')
            callbacks.append(lr_logging_callback)

        if self.clearml_task:
            callbacks.append(S3Callback(self.clearml_task.name))

        return callbacks

    def run_experiment(self):
        # if os.path.exists(self.checkpoint_dir):
        #     shutil.rmtree(self.checkpoint_dir)

        # os.makedirs(os.path.join(self.checkpoint_dir,'logs'), exist_ok=True)

        pl.seed_everything(self.seed)

        train_dataset = FlightDataset(self.datapath, self.features, self.label, self.mode3_column,
                                      self.callsign_column, "train", self.transforms, self.time_encoding_dims)
        valid_dataset = FlightDataset(self.datapath, self.features, self.label, self.mode3_column,
                                      self.callsign_column, "valid", self.transforms, self.time_encoding_dims)

        y_padding = train_dataset.labels_map['pad']
        callsign_padding = train_dataset.CALLSIGN_CHAR2IDX['_']
        mode3_padding = train_dataset.MODE3_CHAR2IDX['_']
        train_loader = DataLoader(train_dataset, collate_fn=lambda x: default_collate(x, y_padding, mode3_padding, callsign_padding),
                                  batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        valid_loader = DataLoader(valid_dataset, collate_fn=lambda x: default_collate(x, y_padding, mode3_padding, callsign_padding),
                                  batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        class_weights = {
            'label_segment_count': train_dataset.get_class_weights('label_segment_count'),
            'label_point_count': train_dataset.get_class_weights('label_point_count'),
            'None': train_dataset.get_class_weights('None')
        }
        # for batch in train_loader:
        #     print(batch[0].shape)
        #     print(batch[1])
        #     # print(batch[2])
        #     break
        # -2 for n_class because we have two special tokens

        labels_map = train_dataset.labels_map
        n_callsign_tokens = len(train_dataset.CALLSIGN_CHAR2IDX)
        n_mode3_tokens = len(train_dataset.MODE3_CHAR2IDX)
        n_classes = train_dataset.n_classes
        distributed = self.n_gpu > 1
        if self.clearml_task:
            if self.weight_by != 'None':
                self.clearml_task.connect_configuration({str(i): val for i, val in enumerate(
                    class_weights[self.weight_by].cpu().numpy())}, name='Class Weights')
            self.clearml_task.connect_configuration(
                labels_map, name='Labels Map')

            metas = {'Train': train_dataset.metadata.copy(
            ), 'Valid': valid_dataset.metadata.copy()}
            for meta in metas.keys():
                for key in ['labels', 'length', 'track_ids']:
                    metas[meta].pop(key)
                self.clearml_task.connect_configuration(
                    metas[meta], name='{} Metadata'.format(meta))

        # load from checkpoint

        if cfg['train']['resume_from_checkpoint']:
            s3_utils = S3Utils(aip_cfg.s3.bucket,
                               aip_cfg.s3.model_artifact_path)
            model_path = os.path.join(
                cfg['train']['checkpoint_dir'], 'latest_model.ckpt')
            s3_utils.s3_download_file(os.path.join(
                self.clearml_task.name, 'latest_model.ckpt'), model_path)
            model = Seq2Seq.load_from_checkpoint(checkpoint_path=model_path)
        else:

            model = Seq2Seq(self.learning_rate,
                            self.lr_schedule,
                            self.n_features,
                            self.d_model,
                            self.dim_feedforward,
                            self.nhead,
                            self.num_encoder_layers,
                            self.num_decoder_layers,
                            self.enc_dropout,
                            self.dec_dropout,
                            self.input_dropout,
                            self.transformer_activation,
                            self.id_embed_dim,
                            n_mode3_tokens,
                            self.n_mode3_token_embedding,
                            self.n_mode3_token_layers,
                            n_callsign_tokens,
                            self.n_callsign_token_embedding,
                            self.n_callsign_token_layers,
                            n_classes,
                            class_weights,
                            self.weight_by,
                            labels_map,
                            distributed)
        callbacks = self._get_callbacks()
        logger = self._get_logger()

        trainer = pl.Trainer(
            gpus=self.n_gpu,
            accelerator=self.accelerator if self.n_gpu > 1 else None,
            callbacks=callbacks,
            logger=logger,
            max_epochs=self.n_epochs,
            default_root_dir=self.checkpoint_dir,
            log_every_n_steps=self.log_every_n_steps
        )

        if self.auto_lr:
            lr_finder = trainer.tuner.lr_find(
                model, train_loader, valid_loader)
            new_lr = lr_finder.suggestion()
            model.learning_rate = new_lr
            print('Found a starting LR of {}'.format(new_lr))
        trainer.fit(model, train_loader, valid_loader)

    @staticmethod
    def add_experiment_args(parent_parser):

        def get_unnested_dict(d, root=''):
            unnested_dict = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    unnested_dict.update(
                        get_unnested_dict(value, root+key+'_'))
                else:
                    unnested_dict[root+key] = value
            return unnested_dict
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        unnested_args = get_unnested_dict(cfg)
        for key, value in unnested_args.items():
            # only parse int,float or str
            if isinstance(value, (int, str, float)):
                # do not parse transforms and lr schedule as we want them as nested dicts
                if 'transforms' not in key and 'lr_schedule' not in key:
                    parser.add_argument('--'+key, default=value)

        return parser

    @staticmethod
    def create_torchscript_model(model_name):
        model = Seq2Seq.load_from_checkpoint(os.path.join(
            cfg['train']['checkpoint_dir'], model_name))

        model.eval()

        # remove_empty_attributes(model)
        # print(vars(model._modules['input_mapper']))
        # print('These attributes should have been removed', remove_attributes)
        script = model.to_torchscript()
        torch.jit.save(script, os.path.join(
            cfg['train']['checkpoint_dir'], "model.pt"))

    @staticmethod
    def create_torchscript_cpu_model(model_name):
        model = Seq2Seq.load_from_checkpoint(os.path.join(
            cfg['train']['checkpoint_dir'], model_name))

        model.to('cpu')
        model.eval()

        # remove_empty_attributes(model)
        # print(vars(model._modules['input_mapper']))
        # print('These attributes should have been removed', remove_attributes)
        script = model.to_torchscript()
        torch.jit.save(script, os.path.join(
            cfg['train']['checkpoint_dir'], "model_cpu.pt"))


def remove_empty_attributes(module):
    remove_attributes = []
    for key, value in vars(module).items():
        if value is None:

            if key == 'trainer' or '_' == key[0]:
                remove_attributes.append(key)
        elif key == '_modules':
            for mod in value.keys():

                remove_empty_attributes(value[mod])
    print('To be removed', remove_attributes)
    for key in remove_attributes:

        delattr(module, key)


# class CustomCheckpoint(ModelCheckpoint):
#     CHECKPOINT_JOIN_CHAR = "-"
#     CHECKPOINT_NAME_LAST = "last"
#     FILE_EXTENSION = ".ckpt"
#     STARTING_VERSION = 1


#     def __init__(
#         self,
#         task_name = None,
#         dirpath: Optional[Union[str, Path]] = None,
#         filename: Optional[str] = None,
#         monitor: Optional[str] = None,
#         verbose: bool = False,
#         save_last: Optional[bool] = None,
#         save_top_k: Optional[int] = None,
#         save_weights_only: bool = False,
#         mode: str = "min",
#         auto_insert_metric_name: bool = True,
#         every_n_train_steps: Optional[int] = None,
#         every_n_val_epochs: Optional[int] = None,
#         period: Optional[int] = None,
#     ):
#         super().__init__(
#             dirpath,
#             filename,
#             monitor,
#             verbose,
#             save_last,
#             save_top_k,
#             save_weights_only,
#             mode,
#             auto_insert_metric_name,
#             every_n_train_steps,
#             every_n_val_epochs,
#             period,
#         )
#         S3_PATH = os.path.join(aip_cfg.s3.model_artifact_path,task_name)
#         latest_model_name = 'latest_model.ckpt'
#         best_model_name = 'best_model.ckpt'


#     def _save_model(self, trainer: 'pl.Trainer', filepath: str) -> None:
#         try:
#             if trainer.training_type_plugin.rpc_enabled:
#                 # RPCPlugin manages saving all model states
#                 # TODO: the rpc plugin should wrap trainer.save_checkpoint
#                 # instead of us having to do it here manually
#                 trainer.training_type_plugin.rpc_save_model(trainer, self._do_save, filepath)
#             else:
#                 self._do_save(trainer, filepath)
#         except:
#             self._do_save(trainer, filepath)

#         # call s3 function here to upload file to s3 using filepath
#         # self.clearml_task.upload_file(filepath,'https://ecs.dsta.ai/bert_finetune_lm/artifact/saved_model.ckpt')

#         # folder = 'uncased' if self.use_uncased else 'cased'

#         print('uploading model checkpoint to S3...')
#         s3_utils = S3Utils(aip_cfg.s3.bucket,aip_cfg.s3.model_artifact_path)


#         s3_utils.s3_upload_file(filepath,os.path.join(S3_PATH,self.latest_model_name))

#         best_model_path = self.best_model_path
#         print("\nBEST MODEL PATH: ", best_model_path, "\n")


#         print('uploading best model checkpoint to S3...')

#         s3_utils.s3_upload_file(best_model_path,os.path.join(S3_PATH,self.best_model_name))
