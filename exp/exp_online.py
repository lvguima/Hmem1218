import copy

import numpy as np
from tqdm import tqdm

from data_provider.data_factory import data_provider, get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Recent
from exp.exp_main import Exp_Main
from models.OneNet import OneNet, Model_Ensemble
from util.buffer import Buffer
from util.metrics import metric, update_metrics, calculate_metrics
from util.online_curve import compute_step_mse, save_online_curve_csv
import torch
import torch.nn.functional as F
from torch import optim, nn
import torch.distributed as dist

import os
import time

import warnings

from util.tools import test_params_flop

warnings.filterwarnings('ignore')

transformers = ['Autoformer', 'Transformer', 'Informer']

class Exp_Online(Exp_Main):
    def __init__(self, args):
        super().__init__(args)
        self.online_phases = ['test', 'online']
        self.wrap_data_kwargs.update(recent_num=1, gap=self.args.pred_len)

    def _get_data(self, flag, **kwargs):
        if flag in self.online_phases:
            if self.args.leakage:
                data_set = get_dataset(self.args, flag, self.device, wrap_class=self.args.wrap_data_class, **kwargs)
                data_loader = get_dataloader(data_set, self.args, 'online' if flag == 'test' else 'test')
            else:
                data_set = get_dataset(self.args, flag, self.device,
                                       wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                       **self.wrap_data_kwargs, **kwargs)
                data_loader = get_dataloader(data_set, self.args, 'online')
            return data_set, data_loader
        else:
            return super()._get_data(flag, **kwargs)

    def vali(self, vali_data, vali_loader, criterion):
        self.phase = 'val'
        if self.args.leakage or 'val' not in self.online_phases:
            mse = super().vali(vali_data, vali_loader, criterion)
        else:
            if self.args.local_rank <= 0:
                state_dict = copy.deepcopy(self.state_dict())
                mse = self.online(online_data=vali_data, target_variate=None, phase='val')[0]
                if self.args.local_rank == 0:
                    mse = torch.tensor(mse, device=self.device)
                self.load_state_dict(state_dict, strict=not (hasattr(self.args, 'freeze') and self.args.freeze))
            else:
                mse = torch.tensor(0, device=self.device)
            if self.args.local_rank >= 0:
                dist.all_reduce(mse, op=dist.ReduceOp.SUM)
                mse = mse.item()
        return mse

    def update_valid(self, valid_data=None):
        self.phase = 'online'

        if hasattr(self.args, 'leakage') and self.args.leakage:
            if self.args.model == 'PatchTST':
                valid_data = get_dataset(self.args, 'val', self.device,
                                         wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                         take_post=self.args.pred_len - 1, **self.wrap_data_kwargs)
                self.online_information_leakage_PatchTST(valid_data, None, 'online', True)
            else:
                valid_data = get_dataset(self.args, 'val', self.device, wrap_class=self.args.wrap_data_class,
                                         take_pre=True, take_post=self.args.pred_len - 1, **self.wrap_data_kwargs)
                self.online_information_leakage(valid_data, None, 'online', True)
            return []

        if valid_data is None or not isinstance(valid_data, Dataset_Recent):
            valid_data = get_dataset(self.args, 'val', self.device,
                                     wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                     take_post=self.args.pred_len - 1, **self.wrap_data_kwargs)
        valid_loader = get_dataloader(valid_data, self.args, 'online')
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        self.model.train()
        predictions = []
        for i, (recent_batch, current_batch) in enumerate(tqdm(valid_loader, mininterval=10)):
            self._update_online(recent_batch, criterion, model_optim, scaler)
            if self.args.do_predict:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.forward(current_batch)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                predictions.append(outputs.detach().cpu().numpy())
                self.model.train()
        return predictions

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        if batch[0].dim() == 3:
            return self._update(batch, criterion, optimizer, scaler)
        else:
            batch = [b[0] for b in batch]
            if not isinstance(optimizer, tuple):
                optimizer = (optimizer,)
            for optim in optimizer:
                optim.zero_grad()
            outputs = self.forward(batch)
            batch_y = batch[1]
            if not self.args.pin_gpu:
                batch_y = batch_y.to(self.device)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            loss = 0
            H = batch_y.shape[1]
            for i in range(H):
                loss += criterion(outputs[i, :H-i], batch_y[i, :H-i])
            if self.args.use_amp:
                scaler.scale(loss).backward()
                for optim in optimizer:
                    scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                for optim in optimizer:
                    optim.step()
            return loss, outputs

    def online(self, online_data=None, target_variate=None, phase='test', show_progress=False):
        self.phase = phase
        if hasattr(self.args, 'leakage') and self.args.leakage:
            if self.args.model == 'PatchTST':
                return self.online_information_leakage_PatchTST(online_data, target_variate, phase, show_progress)
            else:
                return self.online_information_leakage(online_data, target_variate, phase, show_progress)
        if online_data is None:
            online_data = get_dataset(self.args, phase, self.device,
                                      wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                      **self.wrap_data_kwargs)
        # online_loader_initial = get_dataloader(online_data.dataset, self.args, flag='online')
        online_loader = get_dataloader(online_data, self.args, flag='online')

        if self.args.do_predict:
            predictions = []
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE']}
        step_mse = []

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        if phase == 'test' and hasattr(self.args, 'debug') and self.args.debug:
            import tensorboardX as tensorboard
            import shutil
            log_dir = f'run/{self.args.online_method}_{self.args.dataset}_{self.args.seq_len}_{self.args.pred_len}_' \
                      f'{self.args.learning_rate}_{self.args.online_learning_rate}_{self.args.trigger_threshold}_' \
                      f'{self.args.tune_mode}_' \
                      f'{self.args.bottleneck_dim}_{self.args.penalty}_{self.args.comment}/' \
                      f'{time.strftime("%Y%m%d%H%M", time.localtime())}'
            print(log_dir)
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            self.writer = tensorboard.SummaryWriter(log_dir=log_dir)

        if phase == 'test' or show_progress:
            online_loader = tqdm(online_loader, mininterval=10)
        for i, (recent_data, current_data) in enumerate(online_loader):
            self.model.train()
            loss, _ = self._update_online(recent_data, criterion, model_optim, scaler)
            # assert not torch.isnan(loss)
            self.model.eval()
            with torch.no_grad():
                outputs = self.forward(current_data)
                if self.args.do_predict:
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    predictions.append(outputs.detach().cpu().numpy())
                true = current_data[self.label_position]
                if not self.args.pin_gpu:
                    true = true.to(self.device)
                update_metrics(outputs, true, statistics, target_variate)
                step_mse.extend(compute_step_mse(outputs, true, target_variate))

                if phase == 'test' and hasattr(self.args, 'debug') and self.args.debug:
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    mse = F.mse_loss(outputs, current_data[self.label_position].to(self.device))
                    self.writer.add_scalar('Online/MSE', mse, i)
                    self.writer.add_scalar('Online/avg_MSE', statistics['MSE'] / statistics['total'], i)
                    # print('Online MSE: {:.2f}'.format(mse.item()))
                    # for j in range(current_data[self.label_position].shape[-1]):
                    #     self.writer.add_scalar(f'Online/x_{j}', current_data[self.label_position][0, 0, j], i)

        metrics = calculate_metrics(statistics)
        mse, mae = metrics['MSE'], metrics['MAE']
        rmse = metrics.get('RMSE', 0.0)
        rse = metrics.get('RSE', 0.0)
        r2 = metrics.get('R2', 0.0)
        mape = metrics.get('MAPE', 0.0)
        if phase == 'test':
            print('MSE:{:.6f}, MAE:{:.6f}, RMSE:{:.6f}, RSE:{:.6f}, R2:{:.6f}, MAPE:{:.6f}'.format(mse, mae, rmse, rse, r2, mape))
            if getattr(self.args, 'border_type', None) == 'online' and getattr(self.args, 'local_rank', -1) <= 0:
                window = getattr(self.args, 'rolling_window', 500)
                save_online_curve_csv(self.args, step_mse, window=window)
        if self.args.do_predict:
            return mse, mae, online_data, predictions
        else:
            return mse, mae, online_data

    def online_information_leakage(self, online_data=None, target_variate=None, phase='test', show_progress=False):
        if online_data is None:
            online_data = get_dataset(self.args, phase, self.device, wrap_class=self.args.wrap_data_class,
                                      **self.wrap_data_kwargs)
        online_loader = get_dataloader(online_data, self.args, flag='online')

        if self.args.do_predict:
            predictions = []

        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE']}
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        if phase == 'test' or show_progress:
            online_loader = tqdm(online_loader, mininterval=10)
        self.model.train()
        for i, current_data in enumerate(online_loader):
            loss, outputs = self._update_online(current_data, criterion, model_optim, scaler)
            with torch.no_grad():
                update_metrics(outputs, current_data[self.label_position].to(self.device), statistics, target_variate)
            if self.args.do_predict:
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                predictions.append(outputs.detach().cpu().numpy())

        metrics = calculate_metrics(statistics)
        mse, mae = metrics['MSE'], metrics['MAE']
        rmse = metrics.get('RMSE', 0.0)
        rse = metrics.get('RSE', 0.0)
        r2 = metrics.get('R2', 0.0)
        mape = metrics.get('MAPE', 0.0)
        if phase == 'test':
            print('MSE:{:.6f}, MAE:{:.6f}, RMSE:{:.6f}, RSE:{:.6f}, R2:{:.6f}, MAPE:{:.6f}'.format(mse, mae, rmse, rse, r2, mape))
        if self.args.do_predict:
            return mse, mae, online_data, predictions
        else:
            return mse, mae, online_data

    def online_information_leakage_PatchTST(self, online_data=None, target_variate=None, phase='test', show_progress=False):

        self.phase = phase
        if online_data is None:
            online_data = get_dataset(self.args, phase, self.device, wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                      **self.wrap_data_kwargs)
        # online_loader_initial = get_dataloader(online_data.dataset, self.args, flag='online')
        online_loader = get_dataloader(online_data, self.args, flag='online')

        if self.args.do_predict:
            predictions = []
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE']}

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        if phase == 'test' or show_progress:
            online_loader = tqdm(online_loader, mininterval=10)
        for i, (recent_data, current_data) in enumerate(online_loader):
            self.model.train()
            with torch.no_grad():
                outputs = self.forward(recent_data)
            self.model.eval()
            loss, outputs = self._update_online(current_data, criterion, model_optim, scaler)
            # assert not torch.isnan(loss)
            # with torch.no_grad():
                # outputs = self.forward(current_data)
            if self.args.do_predict:
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                predictions.append(outputs.detach().cpu().numpy())
            update_metrics(outputs, current_data[self.label_position].to(self.device), statistics, target_variate)

        metrics = calculate_metrics(statistics)
        mse, mae = metrics['MSE'], metrics['MAE']
        rmse = metrics.get('RMSE', 0.0)
        rse = metrics.get('RSE', 0.0)
        r2 = metrics.get('R2', 0.0)
        mape = metrics.get('MAPE', 0.0)
        if phase == 'test':
            print('MSE:{:.6f}, MAE:{:.6f}, RMSE:{:.6f}, RSE:{:.6f}, R2:{:.6f}, MAPE:{:.6f}'.format(mse, mae, rmse, rse, r2, mape))
        if self.args.do_predict:
            return mse, mae, online_data, predictions
        else:
            return mse, mae, online_data

    def analysis_online(self):
        online_data = get_dataset(self.args, 'test', self.device, wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                  **self.wrap_data_kwargs)
        online_loader = get_dataloader(online_data, self.args, flag='online')
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        times_update = []
        times_infer = []
        print('GPU Mem:', torch.cuda.max_memory_allocated())
        for i, (recent_data, current_data) in enumerate(online_loader):
            start_time = time.time()
            self.model.train()
            recent_data = [d.to(self.device) for d in recent_data]
            loss, _ = self._update_online(recent_data, criterion, model_optim, scaler)
            if i > 10:
                times_update.append(time.time() - start_time)
            self.model.eval()
            with torch.no_grad():
                start_time = time.time()
                current_data = [d.to(self.device) for d in current_data]
                self.forward(current_data)
            # if i == 0:
            #     print('New GPU Mem:', torch.cuda.memory_allocated())
            if i > 10:
                times_infer.append(time.time() - start_time)
            if i == 50:
                break
        print('Final GPU Mem:', torch.cuda.max_memory_allocated())
        times_update = (sum(times_update) - min(times_update) - max(times_update)) / (len(times_update) - 2)
        times_infer = (sum(times_infer) - min(times_infer) - max(times_infer)) / (len(times_infer) - 2)
        print('Update Time:', times_update)
        print('Infer Time:', times_infer)
        print('Latency:', times_update + times_infer)
        test_params_flop(self.model, (1, self.args.seq_len, self.args.enc_in))

    def predict(self, path, setting, load=False):
        self.update_valid()
        res = self.online()
        np.save(f'./results/{self.args.dataset}_{self.args.seq_len}_{self.args.pred_len}'
                               f'_{self.args.model}_{setting[-1]}_pred.npy', np.vstack(res[-1]))
        return None, None


class Exp_ER(Exp_Online):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = Buffer(500, self.device)
        self.count = 0

    def train_loss(self, criterion, batch, outputs):
        loss = super().train_loss(criterion, batch, outputs)
        if not self.buffer.is_empty():
            buff = self.buffer.get_data(8)
            out = self.forward(buff[:-1])
            if isinstance(outputs, (tuple, list)):
                out = out[0]
            loss += 0.2 * criterion(out, buff[1])
        return loss

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        loss, outputs = self._update(batch, criterion, optimizer, scaler=None)
        idx = self.count + torch.arange(batch[1].size(0)).to(self.device)
        self.count += batch[1].size(0)
        self.buffer.add_data(*(batch + [idx]))
        return loss, outputs


class Exp_DERpp(Exp_ER):

    def train_loss(self, criterion, batch, outputs):
        loss = Exp_Online.train_loss(self, criterion, batch, outputs)
        if not self.buffer.is_empty():
            buff = self.buffer.get_data(8)
            out = self.forward(buff[:-1])
            if isinstance(outputs, (tuple, list)):
                out = out[0]
            loss += 0.2 * criterion(buff[-1], out)
        return loss

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        loss, outputs = Exp_Online._update_online(self, batch, criterion, optimizer, scaler)
        self.count += batch[1].size(0)
        if isinstance(outputs, (tuple, list)):
            self.buffer.add_data(*(batch + [outputs[0]]))
        else:
            self.buffer.add_data(*(batch + [outputs]))
        return loss, outputs


class Exp_FSNet(Exp_Online):
    def __init__(self, args):
        super().__init__(args)

    def _update(self, *args, **kwargs):
        ret = super()._update(*args, **kwargs)
        if hasattr(self.model, 'store_grad'):
            self.model.store_grad()
        return ret

    def vali(self, *args, **kwargs):
        if not hasattr(self.model, 'try_trigger_'):
            return super().vali(*args, **kwargs)
        else:
            self.model.try_trigger_(True)
            ret = super().vali(*args, **kwargs)
            self.model.try_trigger_(False)
            return ret

    def online(self, *args, **kwargs):
        if not hasattr(self.model, 'try_trigger_'):
            return super().online(*args, **kwargs)
        else:
            self.model.try_trigger_(True)
            ret = super().online(*args, **kwargs)
            self.model.try_trigger_(False)
            return ret

    def analysis_online(self):
        if hasattr(self.model, 'try_trigger_'):
            self.model.try_trigger_(True)
        return super().analysis_online()


class Exp_OneNet(Exp_FSNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opt_w = optim.Adam([self.model.weight], lr=self.args.learning_rate_w)
        self.opt_bias = optim.Adam(self.model.decision.parameters(), lr=self.args.learning_rate_bias)
        self.bias = torch.zeros(self.args.enc_in, device=self.model.weight.device)

    def _select_optimizer(self, filter_frozen=True, return_self=True, model=None):
        if model is None or isinstance(model, OneNet):
            return super()._select_optimizer(filter_frozen, return_self, model=self.model.backbone)
        return super()._select_optimizer(filter_frozen, return_self, model=model)

    def state_dict(self, *args, **kwargs):
        destination = super().state_dict(*args, **kwargs)
        destination['opt_w'] = self.opt_w.state_dict()
        destination['opt_bias'] = self.opt_bias.state_dict()
        return destination

    # def load_state_dict(self, state_dict, model=None):
    #     self.model.bias.data = state_dict['model']['bias']
    #     return super().load_state_dict(state_dict, model)

    def _build_model(self, model=None, framework_class=None):
        if self.args.model not in ['TCN', 'FSNet', 'TCN_Ensemble', 'FSNet_Ensemble']:
            framework_class = [Model_Ensemble, OneNet]
        else:
            framework_class = OneNet
        return super()._build_model(model, framework_class=framework_class)

    def train_loss(self, criterion, batch, outputs):
        return super().train_loss(criterion, batch, outputs[1]) + super().train_loss(criterion, batch, outputs[2])

    def vali(self, vali_data, vali_loader, criterion):
        self.bias = torch.zeros(self.args.enc_in, device=self.model.weight.device)
        ret = super().vali(vali_data, vali_loader, criterion)
        self.phase = None
        return ret

    def update_valid(self, valid_data=None):
        self.bias = torch.zeros(self.args.enc_in, device=self.model.weight.device)
        return super().update_valid(valid_data)

    def forward(self, batch):
        b, t, d = batch[1].shape
        if hasattr(self, 'phase') and self.phase in self.online_phases:
            weight = self.model.weight.view(1, 1, -1).repeat(b, t, 1)
            bias = self.bias.view(-1, 1, d)
            loss1 = F.sigmoid(weight + bias.repeat(1, t, 1)).view(b, t, d)
        else:
            loss1 = F.sigmoid(self.model.weight).view(1, 1, -1)
            loss1 = loss1.repeat(b, t, 1)
        batch = batch + [loss1, 1 - loss1]
        return super().forward(batch)

    def _update(self, batch, criterion, optimizer, scaler=None):
        batch_y = batch[1]
        b, t, d = batch_y.shape

        loss, (outputs, y1, y2) = super()._update(batch, criterion, optimizer, scaler)

        loss_w = criterion(outputs, batch_y)
        loss_w.backward()
        self.opt_w.step()
        self.opt_w.zero_grad()

        y1_w, y2_w = y1.detach(), y2.detach()
        true_w = batch_y.detach()
        loss1 = F.sigmoid(self.model.weight).view(1, 1, -1)
        loss1 = loss1.repeat(b, t, 1)
        inputs_decision = torch.cat([loss1 * y1_w, (1 - loss1) * y2_w, true_w], dim=1)
        bias = self.model.decision(inputs_decision.permute(0, 2, 1)).view(b, 1, -1)
        weight = self.model.weight.view(1, 1, -1).repeat(b, t, 1)
        loss1 = F.sigmoid(weight + bias.repeat(1, t, 1))
        loss2 = 1 - loss1
        loss_bias = criterion(loss1 * y1_w + loss2 * y2_w, true_w)
        loss_bias.backward()
        self.opt_bias.step()
        self.opt_bias.zero_grad()

        return loss / 2, outputs

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        batch_y = batch[1]
        b, t, d = batch_y.shape

        loss, (outputs, y1, y2) = super()._update(batch, criterion, optimizer, scaler)

        y1_w, y2_w = y1.detach(), y2.detach()
        loss1 = F.sigmoid(self.model.weight).view(1, 1, -1).repeat(b, t, 1)
        inputs_decision = torch.cat([loss1 * y1_w, (1 - loss1) * y2_w, batch_y], dim=1)
        self.bias = self.model.decision(inputs_decision.permute(0, 2, 1))
        weight = self.model.weight.view(1, 1, -1).repeat(b, t, 1)
        bias = self.bias.view(b, 1, -1)
        loss1 = F.sigmoid(weight + bias.repeat(1, t, 1))
        loss2 = 1 - loss1

        outputs_bias = loss1 * y1_w + loss2 * y2_w
        loss_bias = criterion(outputs_bias, batch_y)
        loss_bias.backward()
        self.opt_bias.step()
        self.opt_bias.zero_grad()

        loss1 = F.sigmoid(self.model.weight).view(1, 1, -1)
        loss1 = loss1.repeat(b, t, 1)
        loss_w = criterion(loss1 * y1_w + (1 - loss1) * y2_w, batch_y)
        loss_w.backward()
        self.opt_w.step()
        self.opt_w.zero_grad()
        return loss / 2, outputs


class Exp_ACL(Exp_Online):
    def __init__(self, args):
        super().__init__(args)
        self.buffer_size = getattr(args, 'acl_buffer_size', 500)
        self.soft_buffer_size = getattr(args, 'acl_soft_buffer_size', 50)
        self.alpha = getattr(args, 'acl_alpha', 0.2)
        self.beta = getattr(args, 'acl_beta', 0.2)
        self.gamma = getattr(args, 'acl_gamma', 0.2)
        self.task_interval = getattr(args, 'acl_task_interval', 200)

    def _init_components(self):
        if hasattr(self, 'memory_buffer'):
            return
        from util.acl_utils import ReservoirBuffer, SoftBuffer

        self.memory_buffer = ReservoirBuffer(capacity=self.buffer_size, device=self.device)
        self.soft_buffer = SoftBuffer(capacity=self.soft_buffer_size, device=self.device)

        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        self.step_count = 0

    def update_valid(self, valid_data=None):
        self._init_components()
        return super().update_valid(valid_data)

    def online(self, online_data=None, target_variate=None, phase='test', show_progress=False):
        self._init_components()
        return super().online(online_data, target_variate, phase, show_progress)

    def _forward_with_model(self, model, batch):
        if not self.args.pin_gpu:
            batch = [
                batch[i].to(self.device) if isinstance(batch[i], torch.Tensor) and i != self.label_position else batch[i]
                for i in range(len(batch))
            ]
        inp = self._process_batch(batch)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(*inp)
        else:
            outputs = model(*inp)
        return outputs

    def _split_outputs(self, outputs):
        if isinstance(outputs, (tuple, list)):
            pred = outputs[0]
            enc_out = outputs[1] if len(outputs) > 1 else None
        else:
            pred = outputs
            enc_out = None
        return pred, enc_out

    def _concat_optional(self, tensors):
        if not tensors or all(t is None for t in tensors):
            return None
        keep = [t for t in tensors if t is not None]
        if not keep:
            return None
        return torch.cat(keep, dim=0)

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        if batch[0].dim() != 3:
            return super()._update_online(batch, criterion, optimizer, scaler)

        self.model.train()
        optimizer.zero_grad()

        outputs = self._forward_with_model(self.model, batch)
        pred, enc_out = self._split_outputs(outputs)

        batch_y = batch[1]
        if not self.args.pin_gpu:
            batch_y = batch_y.to(self.device)
        loss_current = criterion(pred, batch_y)

        loss_memory = torch.tensor(0.0, device=self.device)
        loss_feature = torch.tensor(0.0, device=self.device)

        mem_samples = self.memory_buffer.sample(batch_size=min(16, len(self.memory_buffer)))
        soft_samples = self.soft_buffer.get_data()
        replay_samples = []
        if mem_samples is not None:
            replay_samples.append(mem_samples)
        if soft_samples is not None:
            replay_samples.append(soft_samples)

        if replay_samples:
            r_x = torch.cat([s[0] for s in replay_samples], dim=0)
            r_y = torch.cat([s[1] for s in replay_samples], dim=0)
            r_z = torch.cat([s[2] for s in replay_samples], dim=0)
            r_x_mark = self._concat_optional([s[3] for s in replay_samples])
            r_y_mark = self._concat_optional([s[4] for s in replay_samples])

            replay_batch = [r_x, r_y]
            if r_x_mark is not None or r_y_mark is not None:
                replay_batch.append(r_x_mark)
                if r_y_mark is not None:
                    replay_batch.append(r_y_mark)

            r_outputs = self._forward_with_model(self.model, replay_batch)
            r_pred, r_enc = self._split_outputs(r_outputs)
            loss_memory = self.alpha * criterion(r_pred, r_y)
            if r_enc is not None and r_z is not None:
                loss_feature = self.beta * F.mse_loss(r_enc, r_z)

        loss_hint = torch.tensor(0.0, device=self.device)
        if enc_out is not None:
            with torch.no_grad():
                teacher_outputs = self._forward_with_model(self.teacher_model, batch)
                _, teacher_enc = self._split_outputs(teacher_outputs)
            if teacher_enc is not None:
                loss_hint = self.gamma * F.mse_loss(enc_out, teacher_enc)

        total_loss = loss_current + loss_memory + loss_feature + loss_hint

        if self.args.use_amp and scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        batch_x = batch[0]
        batch_x_mark = batch[2] if len(batch) > 2 else None
        batch_y_mark = batch[3] if len(batch) > 3 else None

        feature_tensor = enc_out if enc_out is not None else pred.detach()
        with torch.no_grad():
            raw_losses = torch.mean((pred - batch_y) ** 2, dim=(1, 2))
            self.soft_buffer.update(batch_x, batch_y, feature_tensor, raw_losses, x_mark=batch_x_mark, y_mark=batch_y_mark)
            self.memory_buffer.update(batch_x, batch_y, feature_tensor, x_mark=batch_x_mark, y_mark=batch_y_mark)

        self.step_count += 1
        if self.step_count % self.task_interval == 0:
            self.teacher_model.load_state_dict(self.model.state_dict())
            self.soft_buffer.clear()

        return total_loss, pred


class Exp_CLSER(Exp_Online):
    def __init__(self, args):
        super().__init__(args)
        self.buffer_size = getattr(args, 'clser_buffer_size', 500)
        self.reg_weight = getattr(args, 'clser_reg_weight', 0.15)

    def _init_components(self):
        if hasattr(self, 'clser_buffer'):
            return
        from util.clser_utils import CLSER_Manager, CLSER_Buffer

        self.clser_manager = CLSER_Manager(self.model, self.args, self.device)
        self.clser_buffer = CLSER_Buffer(capacity=self.buffer_size, device=self.device)

    def update_valid(self, valid_data=None):
        self._init_components()
        return super().update_valid(valid_data)

    def online(self, online_data=None, target_variate=None, phase='test', show_progress=False):
        self._init_components()
        return super().online(online_data, target_variate, phase, show_progress)

    def _forward_with_model(self, model, batch):
        if not self.args.pin_gpu:
            batch = [
                batch[i].to(self.device) if isinstance(batch[i], torch.Tensor) and i != self.label_position else batch[i]
                for i in range(len(batch))
            ]
        inp = self._process_batch(batch)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(*inp)
        else:
            outputs = model(*inp)
        return outputs

    def _split_outputs(self, outputs):
        if isinstance(outputs, (tuple, list)):
            pred = outputs[0]
        else:
            pred = outputs
        return pred

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        if batch[0].dim() != 3:
            return super()._update_online(batch, criterion, optimizer, scaler)

        self.model.train()
        optimizer.zero_grad()

        outputs = self._forward_with_model(self.model, batch)
        pred = self._split_outputs(outputs)

        batch_y = batch[1]
        if not self.args.pin_gpu:
            batch_y = batch_y.to(self.device)
        loss_current = criterion(pred, batch_y)

        loss_consistency = torch.tensor(0.0, device=self.device)
        if not self.clser_buffer.is_empty():
            buffer_x, buffer_y, buffer_x_mark, buffer_y_mark = self.clser_buffer.sample(
                batch_size=min(self.args.batch_size, len(self.clser_buffer.buffer))
            )
            if buffer_x is not None:
                buffer_batch = [buffer_x, buffer_y]
                if buffer_x_mark is not None or buffer_y_mark is not None:
                    buffer_batch.append(buffer_x_mark)
                    if buffer_y_mark is not None:
                        buffer_batch.append(buffer_y_mark)
                loss_consistency = self.clser_manager.compute_consistency_loss(
                    buffer_batch, forward_fn=self._forward_with_model
                )
                loss_consistency = self.clser_manager.reg_weight * loss_consistency

        total_loss = loss_current + loss_consistency

        if self.args.use_amp and scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        batch_x = batch[0]
        batch_x_mark = batch[2] if len(batch) > 2 else None
        batch_y_mark = batch[3] if len(batch) > 3 else None
        with torch.no_grad():
            self.clser_buffer.update(batch_x, batch_y, x_mark=batch_x_mark, y_mark=batch_y_mark)

        self.clser_manager.maybe_update_ema_models()
        return total_loss, pred


class Exp_MIR(Exp_Online):
    def __init__(self, args):
        super().__init__(args)
        self.buffer_size = getattr(args, 'mir_buffer_size', 500)
        self.mir_subsample = getattr(args, 'mir_subsample', 500)
        self.mir_k = getattr(args, 'mir_k', 50)

    def _init_components(self):
        if hasattr(self, 'mir_buffer'):
            return
        from util.mir_utils import MIR_Buffer

        self.mir_buffer = MIR_Buffer(
            buffer_size=self.buffer_size,
            device=self.device,
            args=self.args,
        )

    def update_valid(self, valid_data=None):
        self._init_components()
        return super().update_valid(valid_data)

    def online(self, online_data=None, target_variate=None, phase='test', show_progress=False):
        self._init_components()
        return super().online(online_data, target_variate, phase, show_progress)

    def _forward_with_model(self, model, batch):
        if not self.args.pin_gpu:
            batch = [
                batch[i].to(self.device) if isinstance(batch[i], torch.Tensor) and i != self.label_position else batch[i]
                for i in range(len(batch))
            ]
        inp = self._process_batch(batch)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(*inp)
        else:
            outputs = model(*inp)
        return outputs

    def _split_outputs(self, outputs):
        if isinstance(outputs, (tuple, list)):
            pred = outputs[0]
        else:
            pred = outputs
        return pred

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        if batch[0].dim() != 3:
            return super()._update_online(batch, criterion, optimizer, scaler)

        self.model.train()
        optimizer.zero_grad()

        outputs = self._forward_with_model(self.model, batch)
        pred = self._split_outputs(outputs)

        batch_y = batch[1]
        if not self.args.pin_gpu:
            batch_y = batch_y.to(self.device)
        loss_current = criterion(pred, batch_y)

        if self.args.use_amp and scaler is not None:
            scaler.scale(loss_current).backward()
        else:
            loss_current.backward()

        loss_mir = torch.tensor(0.0, device=self.device)
        if not self.mir_buffer.is_empty():
            mir_batch = self.mir_buffer.get_data_with_mir(
                self.model,
                criterion,
                batch_size=min(16, self.buffer_size),
                forward_fn=self._forward_with_model,
            )
            if mir_batch is not None and len(mir_batch) > 0:
                mir_outputs = self._forward_with_model(self.model, mir_batch)
                mir_pred = self._split_outputs(mir_outputs)
                mir_y = mir_batch[1]
                loss_mir = criterion(mir_pred, mir_y)
                if self.args.use_amp and scaler is not None:
                    scaler.scale(loss_mir).backward()
                else:
                    loss_mir.backward()

        if self.args.use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        batch_x = batch[0]
        batch_x_mark = batch[2] if len(batch) > 2 else None
        batch_y_mark = batch[3] if len(batch) > 3 else None
        with torch.no_grad():
            buffer_items = [batch_x, batch_y]
            if batch_x_mark is not None:
                buffer_items.append(batch_x_mark)
            if batch_y_mark is not None:
                buffer_items.append(batch_y_mark)
            self.mir_buffer.add_data(*buffer_items)

        total_loss = loss_current + loss_mir
        return total_loss, pred
