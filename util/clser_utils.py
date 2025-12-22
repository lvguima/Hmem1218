"""CLS-ER utilities: EMA managers and replay buffer."""

from copy import deepcopy
import torch
import torch.nn as nn


class CLSER_Manager:
    """Manage plastic/stable EMA models and consistency regularization."""

    def __init__(self, model, args, device):
        self.model = model
        self.device = device
        self.global_step = 0

        self.reg_weight = getattr(args, 'clser_reg_weight', 0.1)

        self.plastic_update_freq = getattr(args, 'clser_plastic_update_freq', 0.9)
        self.plastic_alpha = getattr(args, 'clser_plastic_alpha', 0.999)

        self.stable_update_freq = getattr(args, 'clser_stable_update_freq', 0.7)
        self.stable_alpha = getattr(args, 'clser_stable_alpha', 0.999)

        self.plastic_model = deepcopy(model).to(device)
        self.stable_model = deepcopy(model).to(device)

        self.plastic_model.eval()
        self.stable_model.eval()
        for p in self.plastic_model.parameters():
            p.requires_grad = False
        for p in self.stable_model.parameters():
            p.requires_grad = False

        self.consistency_loss_fn = nn.MSELoss(reduction='none')

    def compute_consistency_loss(self, buffer_batch, forward_fn=None):
        """Compute student-teacher consistency loss on buffer samples."""
        buffer_x = buffer_batch[0]
        buffer_y = buffer_batch[1]

        with torch.no_grad():
            if forward_fn is None:
                stable_pred = self.stable_model(buffer_x)
                plastic_pred = self.plastic_model(buffer_x)
            else:
                stable_pred = forward_fn(self.stable_model, buffer_batch)
                plastic_pred = forward_fn(self.plastic_model, buffer_batch)

            if isinstance(stable_pred, (tuple, list)):
                stable_pred = stable_pred[0]
            if isinstance(plastic_pred, (tuple, list)):
                plastic_pred = plastic_pred[0]

            stable_error = torch.mean((stable_pred - buffer_y) ** 2, dim=(1, 2))
            plastic_error = torch.mean((plastic_pred - buffer_y) ** 2, dim=(1, 2))

            sel_idx = (stable_error < plastic_error).float().view(-1, 1, 1)
            teacher_pred = sel_idx * stable_pred + (1 - sel_idx) * plastic_pred

        if forward_fn is None:
            student_pred = self.model(buffer_x)
        else:
            student_pred = forward_fn(self.model, buffer_batch)
        if isinstance(student_pred, (tuple, list)):
            student_pred = student_pred[0]

        loss_consistency = torch.mean(self.consistency_loss_fn(student_pred, teacher_pred.detach()))
        return loss_consistency

    def update_plastic_model(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.plastic_alpha)
        with torch.no_grad():
            for ema_param, model_param in zip(self.plastic_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(alpha).add_(model_param.data, alpha=1 - alpha)

    def update_stable_model(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.stable_alpha)
        with torch.no_grad():
            for ema_param, model_param in zip(self.stable_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(alpha).add_(model_param.data, alpha=1 - alpha)

    def maybe_update_ema_models(self):
        self.global_step += 1
        if torch.rand(1).item() < self.plastic_update_freq:
            self.update_plastic_model()
        if torch.rand(1).item() < self.stable_update_freq:
            self.update_stable_model()


class CLSER_Buffer:
    """Reservoir buffer storing (x, y, x_mark, y_mark)."""

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.num_seen = 0

    def update(self, x, y, x_mark=None, y_mark=None):
        batch_size = x.shape[0]
        x = x.detach().cpu()
        y = y.detach().cpu()
        if x_mark is not None:
            x_mark = x_mark.detach().cpu()
        if y_mark is not None:
            y_mark = y_mark.detach().cpu()

        for i in range(batch_size):
            sample = {
                'x': x[i],
                'y': y[i],
                'x_mark': x_mark[i] if x_mark is not None else None,
                'y_mark': y_mark[i] if y_mark is not None else None,
            }

            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                idx = torch.randint(0, self.num_seen + 1, (1,)).item()
                if idx < self.capacity:
                    self.buffer[idx] = sample

            self.num_seen += 1

    def sample(self, batch_size):
        if not self.buffer:
            return None, None, None, None

        sample_size = min(batch_size, len(self.buffer))
        indices = torch.randperm(len(self.buffer))[:sample_size]

        batch_x = []
        batch_y = []
        batch_x_mark = []
        batch_y_mark = []

        for idx in indices:
            sample = self.buffer[idx]
            batch_x.append(sample['x'])
            batch_y.append(sample['y'])
            if sample['x_mark'] is not None:
                batch_x_mark.append(sample['x_mark'])
            if sample['y_mark'] is not None:
                batch_y_mark.append(sample['y_mark'])

        x = torch.stack(batch_x).to(self.device)
        y = torch.stack(batch_y).to(self.device)
        x_mark = torch.stack(batch_x_mark).to(self.device) if batch_x_mark else None
        y_mark = torch.stack(batch_y_mark).to(self.device) if batch_y_mark else None

        return x, y, x_mark, y_mark

    def is_empty(self):
        return len(self.buffer) == 0
