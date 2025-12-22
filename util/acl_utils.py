"""ACL utilities (ReservoirBuffer and SoftBuffer)."""

import random
import numpy as np
import torch


class ReservoirBuffer:
    """Reservoir sampling buffer storing (x, y, z, x_mark, y_mark)."""

    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.data = []
        self.seen_samples = 0

    def update(self, x, y, z, x_mark=None, y_mark=None):
        """Add a batch of samples to the buffer."""
        batch_size = x.shape[0]
        x = x.detach().cpu()
        y = y.detach().cpu()
        z = z.detach().cpu()
        if x_mark is not None:
            x_mark = x_mark.detach().cpu()
        if y_mark is not None:
            y_mark = y_mark.detach().cpu()

        for i in range(batch_size):
            self.seen_samples += 1
            sample = {
                'x': x[i],
                'y': y[i],
                'z': z[i],
                'x_mark': x_mark[i] if x_mark is not None else None,
                'y_mark': y_mark[i] if y_mark is not None else None,
            }

            if len(self.data) < self.capacity:
                self.data.append(sample)
            else:
                idx = random.randint(0, self.seen_samples - 1)
                if idx < self.capacity:
                    self.data[idx] = sample

    def sample(self, batch_size):
        """Randomly sample a mini-batch."""
        if not self.data:
            return None

        sample_size = min(len(self.data), batch_size)
        samples = random.sample(self.data, sample_size)

        batch_x = torch.stack([s['x'] for s in samples]).to(self.device)
        batch_y = torch.stack([s['y'] for s in samples]).to(self.device)
        batch_z = torch.stack([s['z'] for s in samples]).to(self.device)

        batch_x_mark = None
        if samples[0]['x_mark'] is not None:
            batch_x_mark = torch.stack([s['x_mark'] for s in samples]).to(self.device)

        batch_y_mark = None
        if samples[0]['y_mark'] is not None:
            batch_y_mark = torch.stack([s['y_mark'] for s in samples]).to(self.device)

        return batch_x, batch_y, batch_z, batch_x_mark, batch_y_mark

    def __len__(self):
        return len(self.data)


class SoftBuffer:
    """Short-term buffer that keeps low-loss samples from the latest batch."""

    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.buffer = []

    def update(self, x, y, z, losses, x_mark=None, y_mark=None):
        """Select lowest-loss samples and store them in the buffer."""
        x = x.detach().cpu()
        y = y.detach().cpu()
        z = z.detach().cpu()
        losses = losses.detach().cpu()
        if x_mark is not None:
            x_mark = x_mark.detach().cpu()
        if y_mark is not None:
            y_mark = y_mark.detach().cpu()

        k = min(len(losses), self.capacity)
        if k == 0:
            return

        topk_indices = torch.argsort(losses)[:k]
        self.buffer = []

        for idx in topk_indices:
            sample = {
                'x': x[idx],
                'y': y[idx],
                'z': z[idx],
                'x_mark': x_mark[idx] if x_mark is not None else None,
                'y_mark': y_mark[idx] if y_mark is not None else None,
            }
            self.buffer.append(sample)

    def get_data(self):
        """Return all stored samples as a batch."""
        if not self.buffer:
            return None

        samples = self.buffer
        batch_x = torch.stack([s['x'] for s in samples]).to(self.device)
        batch_y = torch.stack([s['y'] for s in samples]).to(self.device)
        batch_z = torch.stack([s['z'] for s in samples]).to(self.device)

        batch_x_mark = None
        if samples[0]['x_mark'] is not None:
            batch_x_mark = torch.stack([s['x_mark'] for s in samples]).to(self.device)

        batch_y_mark = None
        if samples[0]['y_mark'] is not None:
            batch_y_mark = torch.stack([s['y_mark'] for s in samples]).to(self.device)

        return batch_x, batch_y, batch_z, batch_x_mark, batch_y_mark

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)
