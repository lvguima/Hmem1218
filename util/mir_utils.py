"""MIR utilities (buffer and sampler)."""

from copy import deepcopy
import numpy as np
import torch


def get_grad_vector(model):
    grad_dims = [param.data.numel() for param in model.parameters()]
    grads = torch.zeros(sum(grad_dims))
    if next(model.parameters()).is_cuda:
        grads = grads.cuda()

    cnt = 0
    for param in model.parameters():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1

    return grads, grad_dims


def overwrite_grad(model, new_grad, grad_dims):
    cnt = 0
    for param in model.parameters():
        param.grad = torch.zeros_like(param.data)
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = new_grad[beg: en].contiguous().view(param.data.size())
        param.grad.data.copy_(this_grad)
        cnt += 1


def get_future_step_parameters(model, grad_vector, grad_dims, lr=1.0):
    virtual_model = deepcopy(model)
    overwrite_grad(virtual_model, grad_vector, grad_dims)

    with torch.no_grad():
        for param in virtual_model.parameters():
            if param.grad is not None:
                param.data = param.data - lr * param.grad.data

    return virtual_model


class MIR_Sampler:
    """Select samples that are most interfered by current gradient."""

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.subsample = getattr(args, 'mir_subsample', 500)
        self.k = getattr(args, 'mir_k', 50)

    def compute_interference_scores(self, model, buffer_batch, grad_vector, grad_dims, criterion, forward_fn=None):
        batch_x = buffer_batch[0]
        batch_y = buffer_batch[1]

        with torch.no_grad():
            if forward_fn is None:
                outputs = model(batch_x)
            else:
                outputs = forward_fn(model, buffer_batch)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            loss_pre = torch.mean((outputs - batch_y) ** 2, dim=(1, 2))

        virtual_model = get_future_step_parameters(
            model, grad_vector, grad_dims, lr=self.args.learning_rate
        )
        virtual_model.eval()

        with torch.no_grad():
            if forward_fn is None:
                outputs_post = virtual_model(batch_x)
            else:
                outputs_post = forward_fn(virtual_model, buffer_batch)
            if isinstance(outputs_post, (tuple, list)):
                outputs_post = outputs_post[0]
            loss_post = torch.mean((outputs_post - batch_y) ** 2, dim=(1, 2))

        interference_scores = loss_post - loss_pre
        del virtual_model
        return interference_scores

    def select_samples(self, model, buffer, criterion, batch_size, forward_fn=None):
        buffer_size = min(self.subsample, buffer.num_seen_examples, len(buffer.buffer[0]))
        if buffer_size == 0:
            return None

        indices = np.random.choice(
            min(buffer.num_seen_examples, len(buffer.buffer[0])),
            size=buffer_size,
            replace=False,
        )

        buffer_batch = [attr[indices].to(self.device) for attr in buffer.buffer]

        grad_vector, grad_dims = get_grad_vector(model)
        scores = self.compute_interference_scores(
            model, buffer_batch, grad_vector, grad_dims, criterion, forward_fn=forward_fn
        )

        k = min(batch_size, len(scores))
        _, top_indices = torch.topk(scores, k, largest=True)

        selected_batch = [attr_batch[top_indices] for attr_batch in buffer_batch]
        return selected_batch


class MIR_Buffer:
    """Reservoir buffer with MIR sampling support."""

    def __init__(self, buffer_size, device, args):
        self.buffer_size = buffer_size
        self.device = device
        self.args = args
        self.buffer = []
        self.num_seen_examples = 0
        self.mir_sampler = MIR_Sampler(args, device)

    def init_tensors(self, *batch):
        for attr in batch:
            self.buffer.append(
                torch.zeros((self.buffer_size, *attr.shape[1:]), dtype=torch.float32, device=self.device)
            )

    def add_data(self, *batch):
        if self.num_seen_examples == 0:
            self.init_tensors(*batch)

        for i in range(batch[0].shape[0]):
            if self.num_seen_examples < self.buffer_size:
                index = self.num_seen_examples
            else:
                index = np.random.randint(0, self.num_seen_examples + 1)

            self.num_seen_examples += 1
            if index < self.buffer_size:
                for j, attr in enumerate(batch):
                    self.buffer[j][index] = attr[i].detach().to(self.device)

    def get_data_with_mir(self, model, criterion, batch_size, forward_fn=None):
        if self.num_seen_examples == 0:
            return None
        return self.mir_sampler.select_samples(
            model, self, criterion, batch_size, forward_fn=forward_fn
        )

    def get_data(self, batch_size):
        if self.num_seen_examples == 0:
            return None

        sample_size = min(batch_size, self.num_seen_examples, len(self.buffer[0]))
        indices = np.random.choice(
            min(self.num_seen_examples, len(self.buffer[0])),
            size=sample_size,
            replace=False,
        )

        batch = [attr[indices].to(self.device) for attr in self.buffer]
        return batch

    def is_empty(self):
        return self.num_seen_examples == 0
