import sys, logging
sys.path.append('../')
import random
from pprint import pformat
from collections import deque
from typing import *
import numpy as np
import torch
import pandas as pd


def last_and_valid_max(col:pd.Series):
    '''
    find last not None value and max valid (not None or np.nan) value for each column
    :param col:
    :return:
    '''
    return pd.Series(
        index=[
        'last', 'valid_max', 'exist_nan_value'
    ],
        data=[
        col[~col.isna()].iloc[-1], pd.to_numeric(col, errors='coerce').max(), any(i == 'nan_value' for i in col)
    ])

class Metric_Aggregator(object):
    '''
    aggregate the metric to log
    '''
    def __init__(self):
        self.history = []
    def __call__(self,
                 one_metric : dict):
        one_metric = {k : v for k,v in one_metric.items() if v is not None} # drop pair with None as value
        one_metric = {
            k : (
            "nan_value" if v is np.nan or torch.tensor(v).isnan().item() else v #turn nan to str('nan_value')
            ) for k, v in one_metric.items()
        }
        self.history.append(one_metric)
        logging.info(
            pformat(
                one_metric
            )
        )
    def to_dataframe(self):
        self.df = pd.DataFrame(self.history, dtype=object)
        logging.info("return df with np.nan and None converted by str()")
        return self.df
    def summary(self):
        if 'df' not in self.__dict__:
            logging.info('No df found in Metric_Aggregator, generate now')
            self.to_dataframe()
        logging.info("return df with np.nan and None converted by str()")
        return self.df.apply(last_and_valid_max)

class ModelTrainerCLS():
    def __init__(self, model):
        self.model = model

    def init_or_continue_train(self,
                               train_data,
                               end_epoch_num,
                               criterion,
                               optimizer,
                               scheduler,
                               device,
                               continue_training_path: Optional[str] = None,
                               only_load_model: bool = False,
                               ) -> None:

        model = self.model

        model.to(device)
        model.train()

        # train and update

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        if continue_training_path is not None:
            start_epoch, start_batch = self.load_from_path(continue_training_path, device, only_load_model)
            if (start_epoch is None) or (start_batch is None):
                self.start_epochs, self.end_epochs = 0, end_epoch_num
                self.start_batch = 0
            else:
                batch_num = len(train_data)
                self.start_epochs, self.end_epochs = start_epoch + ((start_batch + 1)//batch_num), end_epoch_num
                self.start_batch = (start_batch + 1) % batch_num
        else:
            self.start_epochs, self.end_epochs = 0, end_epoch_num
            self.start_batch = 0

        logging.info(f'All setting done, train from epoch {self.start_epochs} batch {self.start_batch} to epoch {self.end_epochs}')

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def save_all_state_to_path(self,
                               path: str,
                               epoch: Optional[int] = None,
                               batch: Optional[int] = None,
                               only_model_state_dict: bool = False) -> None:

        save_dict = {
            'epoch_num_when_save': epoch,
            'batch_num_when_save': batch,
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(),
            'torch_random_state': torch.random.get_rng_state(),
            'model_state_dict': self.get_model_params(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'criterion_state_dict': self.criterion.state_dict(),
        } \
            if only_model_state_dict == False else self.get_model_params()

        torch.save(
            save_dict,
            path,
        )

    def load_from_path(self,
                       path: str,
                       device,
                       only_load_model: bool = False
                       ) -> [Optional[int], Optional[int]]:
        'all move to cpu first, then consider to move to GPU again'

        self.model = self.model.to(device)

        load_dict = torch.load(
            path,
        )

        logging.info(f"loading... keys:{load_dict.keys()}, only_load_model:{only_load_model}")

        attr_list = [
            'epoch_num_when_save',
            'batch_num_when_save',
            'random_state',
            'np_random_state',
            'torch_random_state',
            'model_state_dict',
            'optimizer_state_dict',
            'scheduler_state_dict',
            'criterion_state_dict',
        ]

        if all([key_name in load_dict for key_name in attr_list]) :
            # all required key can find in load dict
            # AND only_load_model == False
            if only_load_model == False:
                random.setstate(load_dict['random_state'])
                np.random.set_state(load_dict['np_random_state'])
                torch.random.set_rng_state(load_dict['torch_random_state'].cpu()) # since may map to cuda

                self.model.load_state_dict(
                    load_dict['model_state_dict']
                )
                self.optimizer.load_state_dict(
                    load_dict['optimizer_state_dict']
                )
                if self.scheduler is not None:
                    self.scheduler.load_state_dict(
                        load_dict['scheduler_state_dict']
                    )
                self.criterion.load_state_dict(
                    load_dict['criterion_state_dict']
                )
                logging.info('all state load successful')
                return load_dict['epoch_num_when_save'], load_dict['batch_num_when_save']
            else:
                self.model.load_state_dict(
                    load_dict['model_state_dict'],
                )
                logging.info('only model state_dict load')
                return None, None

        else:  # only state_dict

            if 'model_state_dict' in load_dict:
                self.model.load_state_dict(
                    load_dict['model_state_dict'],
                )
                logging.info('only model state_dict load')
                return None, None
            else:
                self.model.load_state_dict(
                    load_dict,
                )
                logging.info('only model state_dict load')
                return None, None

    def test(self, test_data, device):
        model = self.model
        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0,
            # 'detail_list' : [],
        }

        criterion = self.criterion.to(device)

        with torch.no_grad():
            for batch_idx, (x, target, *additional_info) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())

                # logging.info(list(zip(additional_info[0].cpu().numpy(), pred.detach().cpu().numpy(),
                #                target.detach().cpu().numpy(), )))

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
                # metrics['detail_list'] += list(zip(
                #     additional_info[0].cpu().numpy(),
                #     pred.detach().cpu().numpy(),
                #     predicted.detach().cpu().numpy(),
                #     target.detach().cpu().numpy(),
                # ))
                # logging.info(f"testing, batch_idx:{batch_idx}, acc:{metrics['test_correct']}/{metrics['test_total']}")

        return metrics

    #@resource_check
    def train_one_batch(self, x, labels, device):

        self.model.train()
        self.model.to(device)

        x, labels = x.to(device), labels.to(device)
        self.model.zero_grad()
        log_probs = self.model(x)
        loss = self.criterion(log_probs, labels.long())
        loss.backward()

        # Uncommet this following line to avoid nan loss
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # logging.info(list(zip(additional_info[0].cpu().numpy(), log_probs.detach().cpu().numpy(), labels.detach().cpu().numpy(), )))

        self.optimizer.step()

        batch_loss = (loss.item())

        return batch_loss
        # logging.info(f"training, epoch:{epoch}, batch:{batch_idx},batch_loss:{loss.item()}")

    def train_one_epoch(self, train_data, device):

        batch_loss = []
        for batch_idx, (x, labels, *additional_info) in enumerate(train_data):
            batch_loss.append(self.train_one_batch(x, labels, device))
        one_epoch_loss = sum(batch_loss) / len(batch_loss)

        if self.scheduler is not None:
            self.scheduler.step()

        return one_epoch_loss

    def train(self, train_data, end_epoch_num,
                               criterion,
                               optimizer,
                               scheduler, device,  frequency_save, save_folder_path,
              save_prefix,
              continue_training_path: Optional[str] = None,
              only_load_model: bool = False, ):

        self.init_or_continue_train(
            train_data,
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            continue_training_path,
            only_load_model
        )
        epoch_loss = []
        for epoch in range(self.start_epochs, self.end_epochs):
            one_epoch_loss = self.train_one_epoch(train_data, device)
            epoch_loss.append(one_epoch_loss)
            logging.info(f'train, epoch_loss: {epoch_loss[-1]}')
            if frequency_save != 0 and epoch % frequency_save == frequency_save - 1:
                logging.info(f'saved. epoch:{epoch}')
                self.save_all_state_to_path(
                    epoch=epoch,
                    path=f"{save_folder_path}/{save_prefix}_epoch_{epoch}.pt")

    def train_with_test_each_epoch(self,
                                   train_data,
                                   test_data,
                                   adv_test_data,
                                   end_epoch_num,
                                   criterion,
                                   optimizer,
                                   scheduler,
                                   device,
                                   frequency_save,
                                   save_folder_path,
                                   save_prefix,
                                   continue_training_path: Optional[str] = None,
                                   only_load_model: bool = False,
                                   ):
        agg = Metric_Aggregator()
        self.init_or_continue_train(
            train_data,
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            continue_training_path,
            only_load_model
        )
        epoch_loss = []
        for epoch in range(self.start_epochs, self.end_epochs):
            one_epoch_loss = self.train_one_epoch(train_data, device)
            epoch_loss.append(one_epoch_loss)
            logging.info(f'train_with_test_each_epoch, epoch:{epoch} ,epoch_loss: {epoch_loss[-1]}')

            metrics = self.test(test_data, device)
            metric_info = {
                'epoch': epoch,
                'benign acc': metrics['test_correct'] / metrics['test_total'],
                'benign loss': metrics['test_loss'],
            }
            agg(metric_info)

            adv_metrics = self.test(adv_test_data, device)
            adv_metric_info = {
                'epoch': epoch,
                'ASR': adv_metrics['test_correct'] / adv_metrics['test_total'],
                'backdoor loss': adv_metrics['test_loss'],
            }
            agg(adv_metric_info)

            if frequency_save != 0 and epoch % frequency_save == frequency_save - 1:
                logging.info(f'saved. epoch:{epoch}')
                self.save_all_state_to_path(
                    epoch=epoch,
                    path=f"{save_folder_path}/{save_prefix}_epoch_{epoch}.pt")
            # logging.info(f"training, epoch:{epoch}, batch:{batch_idx},batch_loss:{loss.item()}")
        agg.to_dataframe().to_csv(f"{save_folder_path}/{save_prefix}_df.csv")
        agg.summary().to_csv(f"{save_folder_path}/{save_prefix}_df_summary.csv")

    def train_with_test_each_epoch_v2(self,
                                   train_data,
                                   test_dataloader_dict,
                                   end_epoch_num,
                                   criterion,
                                   optimizer,
                                   scheduler,
                                   device,
                                   frequency_save,
                                   save_folder_path,
                                   save_prefix,
                                   continue_training_path: Optional[str] = None,
                                   only_load_model: bool = False,
                                   ):
        '''
        v2 can feed many test_dataloader, so more easy for test with multiple dataloader
        :param train_data:
        :param test_dataloader_dict: { name : dataloader }

        :param end_epoch_num:
        :param criterion:
        :param optimizer:
        :param scheduler:
        :param device:
        :param frequency_save:
        :param save_folder_path:
        :param save_prefix:
        :param continue_training_path:
        :param only_load_model:
        :return:
        '''
        agg = Metric_Aggregator()
        self.init_or_continue_train(
            train_data,
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            continue_training_path,
            only_load_model
        )
        epoch_loss = []
        for epoch in range(self.start_epochs, self.end_epochs):
            one_epoch_loss = self.train_one_epoch(train_data, device)
            epoch_loss.append(one_epoch_loss)
            logging.info(f'train_with_test_each_epoch, epoch:{epoch} ,epoch_loss: {epoch_loss[-1]}')

            for dl_name, test_dataloader in test_dataloader_dict.items():
                metrics = self.test(test_dataloader, device)
                metric_info = {
                    'epoch': epoch,
                    f'{dl_name} acc': metrics['test_correct'] / metrics['test_total'],
                    f'{dl_name} loss': metrics['test_loss'],
                }
                agg(metric_info)


            if frequency_save != 0 and epoch % frequency_save == frequency_save - 1:
                logging.info(f'saved. epoch:{epoch}')
                self.save_all_state_to_path(
                    epoch=epoch,
                    path=f"{save_folder_path}/{save_prefix}_epoch_{epoch}.pt")
            # logging.info(f"training, epoch:{epoch}, batch:{batch_idx},batch_loss:{loss.item()}")
        agg.to_dataframe().to_csv(f"{save_folder_path}/{save_prefix}_df.csv")
        agg.summary().to_csv(f"{save_folder_path}/{save_prefix}_df_summary.csv")
