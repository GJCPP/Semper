import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Subset
import torch.nn.functional as F
import numpy as np
import os
import random
import copy

save_cache_to_file = True

class ManualVGG11:
    def __init__(self):
        torch.backends.cudnn.deterministic = True
        self.W = {}

        # Conv layers (no bias), Kaiming init
        conv_shapes = [
            (64, 3, 3, 3),        # Block 1
            (128, 64, 3, 3),      # Block 2
            (256, 128, 3, 3), (256, 256, 3, 3),  # Block 3
            (512, 256, 3, 3), (512, 512, 3, 3),  # Block 4
            (512, 512, 3, 3), (512, 512, 3, 3)   # Block 5
        ]
        self.scale=2**14
        for i, shape in enumerate(conv_shapes):
            self.W[f'conv{i+1}'] = torch.randn(*shape) * (2.0 / (shape[1]*shape[2]*shape[3]))**0.5
            self.W[f'conv{i+1}'].requires_grad = False
            self.W[f'conv_q{i+1}']=torch.round(self.W[f'conv{i+1}']*self.scale).to(torch.int64)

        # FC layers
        self.W['fc1'] = torch.randn(512, 512) * (2.0 / 512)**0.5
        self.W['fc2'] = torch.randn(512, 512) * (2.0 / 512)**0.5
        self.W['fc3'] = torch.randn(512, 10) * (2.0 / 512)**0.5
        self.W['fc1_q']=torch.round(self.W['fc1']*self.scale).to(torch.int64)
        self.W['fc2_q']=torch.round(self.W['fc2']*self.scale).to(torch.int64)
        self.W['fc3_q']=torch.round(self.W['fc3']*self.scale).to(torch.int64)

        for k in ['fc1', 'fc2', 'fc3','fc1_q', 'fc2_q', 'fc3_q']:
            self.W[k].requires_grad = False

        self.cache = {}

    def size(self):
        total_params = 0
        for param in self.W.values():
            total_params += param.numel()
        return total_params // 2 # remove quantized version

    def clear_cache(self):
        self.cache = {}

    def save_to_cache(self, key, value):
        if self.cache.get(key) is None:
            self.cache[key] = []
        self.cache[key].append(copy.deepcopy(value))

    def forward(self, x, y):
        #self.input = x
        idx = 1
        self.input_q = (x*self.scale).to(torch.int64)
        x_q=self.input_q
        self.save_to_cache('a_q0', x_q)
        
        one_hot_y = F.one_hot(y, num_classes=10)
        self.save_to_cache('a_q0_label', one_hot_y)

        for block, layers in enumerate([(1, ), (2, ), (3, 4), (5, 6), (7, 8)], start=1):
            #for lid in layers:
            #    x = F.conv2d(x, self.W[f'conv{lid}'], padding=1)
            #    self.cache[f'z{lid}'] = x
            #    x = F.relu(x)
            #x, pool_idx = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
            #self.cache[f'pool{block}'] = (x, pool_idx)


            for lid in layers:
                x_q = F.conv2d(x_q.to(torch.float64), self.W[f'conv_q{lid}'].to(torch.float64), padding=1)
                self.save_to_cache(f'z_q{lid}', x_q.to(torch.int64))
                x_q = x_q.to(torch.int64) // self.scale
                x_q = F.relu(x_q).to(torch.int64)
                self.save_to_cache(f'W_conv_q{lid}', self.W[f'conv_q{lid}'])
                self.save_to_cache(f'a_q{lid}', x_q)

            x_q, pool_q_idx = F.max_pool2d(x_q, kernel_size=2, stride=2, return_indices=True)
            x_q = x_q.to(torch.int64)
            self.save_to_cache(f'pool_q{block}', x_q)
            self.save_to_cache(f'pool_idx_q{block}', pool_q_idx)
            # import random

            # if block == 1:
            #     W = self.cache['W_conv_q1'][-1]  # shape [64, 3, 3, 3]
            #     x_pad = F.pad(self.input_q, (1, 1, 1, 1), mode='constant', value=0)  # [N, 3, H+2, W+2]
            #     z_ref = self.cache['z_q1'][-1]  # [N, 64, H, W]

            #     N, C, H, W_ = self.input_q.shape
            #     OC = W.shape[0]  # 64

            #     K = 20000  # number of random points to sample
            #     errors = 0

            #     for _ in range(K):
            #         # Randomly sample one point
            #         # n = random.randint(0, N - 1)
            #         # oc = random.randint(0, OC - 1)
            #         # i = random.randint(0, H - 1)
            #         # j = random.randint(0, W_ - 1)
            #         n = 0
            #         oc = 0
            #         i = 0
            #         j = 0

            #         # Manually compute conv+ReLU at that location
            #         acc = 0
            #         for c in range(C):
            #             for ki in range(3):
            #                 for kj in range(3):
            #                     xi = x_pad[n, c, i + ki, j + kj].item()
            #                     wi = W[oc, c, ki, kj].item()
            #                     acc += xi * wi
            #                     print(acc,xi,wi)
            #         acc = acc // self.scale
            #         if acc < 0:
            #             acc = 0

            #         actual = z_ref[n, oc, i, j].item()
            #         if acc != actual:
            #             errors += 1
            #         exit(0)
            #     print(f"Checked {K} random positions — {errors} mismatches.")

        #x = x.view(x.size(0), -1)
        #self.cache['flat'] = x
        #z1 = x @ self.W['fc1']
        #self.cache['fc1_z'] = z1
        #a1 = F.relu(z1)
        #z2 = a1 @ self.W['fc2']
        #self.cache['fc2_z'] = z2
        #a2 = F.relu(z2)
        #logits = a2 @ self.W['fc3']
        #self.cache['logits'] = logits

        x_q = x_q.view(x_q.size(0), -1).to(torch.int64)
        z1_q = (x_q.to(torch.float64) @ self.W['fc1_q'].to(torch.float64)).to(torch.int64)
        a1_q = (F.relu(z1_q)/self.scale).to(torch.int64)
        z2_q = (a1_q.to(torch.float64) @ self.W['fc2_q'].to(torch.float64)).to(torch.int64)
        a2_q = (F.relu(z2_q)/self.scale).to(torch.int64)
        z3_q = (a2_q.to(torch.float64) @ self.W['fc3_q'].to(torch.float64)).to(torch.int64)

        self.save_to_cache('flat_q', x_q)
        self.save_to_cache('W_fc1_q', self.W['fc1_q'])
        self.save_to_cache('z1_q', z1_q)
        self.save_to_cache('a1_q', a1_q)
        self.save_to_cache('W_fc2_q', self.W['fc2_q'])
        self.save_to_cache('z2_q', z2_q)
        self.save_to_cache('a2_q', a2_q)
        self.save_to_cache('W_fc3_q', self.W['fc3_q'])
        self.save_to_cache('z3_q', z3_q)

        return z3_q

    def backward_propogation(self, grad_z3_q,  lr=0.01):
        W = self.W
        c = self.cache
        #print("grad err:",torch.mean(torch.abs(grad_logits-grad_z3_q/self.scale)),grad_logits.max(),grad_logits.min())
        # FC3
        #grad_a2 = grad_logits @ W['fc3'].T
        #dW_fc3 = c['fc2_z'].T @ grad_logits
        #with torch.no_grad():
        #    W['fc3'] -= lr * dW_fc3
        self.save_to_cache('grad_z3_q', grad_z3_q)

        grad_a2_q = (grad_z3_q.to(torch.float64) @ W['fc3_q'].T.to(torch.float64)).to(torch.int64)
        dW_fc3_q = (c['a2_q'][-1].T.to(torch.float64) @ grad_z3_q.to(torch.float64)).to(torch.int64) # big value
        #print(dW_fc3_q.shape,dW_fc3.shape,W['fc3_q'].shape)
        with torch.no_grad():
            W['fc3_q'] -= torch.round(lr * torch.round(dW_fc3_q  / self.scale)).to(torch.int64)

        self.save_to_cache('dW_fc3_q', dW_fc3_q)
        self.save_to_cache('grad_a2_q', grad_a2_q)


        # FC2
        #grad_z2 = grad_a2 * (c['fc2_z'] > 0)
        #grad_a1 = grad_z2 @ W['fc2'].T
        #dW_fc2 = c['fc1_z'].T @ grad_z2
        #with torch.no_grad():
        #    W['fc2'] -= lr * dW_fc2
            

        grad_z2_q = torch.round(grad_a2_q * (c['a2_q'][-1] > 0) / self.scale).to(torch.int64)
        grad_a1_q = torch.round(grad_z2_q.to(torch.float64) @ W['fc2_q'].T.to(torch.float64)).to(torch.int64)
        dW_fc2_q = (c['a1_q'][-1].to(torch.float64).T @ grad_z2_q.to(torch.float64)).to(torch.int64) # big value
        with torch.no_grad():
            W['fc2_q'] -= torch.round(lr * torch.round(dW_fc2_q / self.scale)).to(torch.int64)

        self.save_to_cache('grad_z2_q', grad_z2_q)
        self.save_to_cache('dW_fc2_q', dW_fc2_q)
        self.save_to_cache('grad_a1_q', grad_a1_q)

        # FC1
        #grad_z1 = grad_a1 * (c['fc1_z'] > 0)
        #grad_flat = grad_z1 @ W['fc1'].T
        #dW_fc1 = c['flat'].T @ grad_z1
        #with torch.no_grad():
        #    W['fc1'] -= lr * dW_fc1

        grad_z1_q = torch.round(grad_a1_q * (c['a1_q'][-1] > 0) / self.scale).to(torch.int64)
        grad_flat_q = torch.round(grad_z1_q @ W['fc1_q'].T).to(torch.int64)
        dW_fc1_q = (c['flat_q'][-1].to(torch.float64).T @ grad_z1_q.to(torch.float64)).to(torch.int64) # big value
        with torch.no_grad():
            W['fc1_q'] -= torch.round(lr * torch.round(dW_fc1_q / self.scale)).to(torch.int64)

        self.save_to_cache('grad_z1_q', grad_z1_q)
        self.save_to_cache('dW_fc1_q', dW_fc1_q)
        self.save_to_cache('grad_flat_q', grad_flat_q)

        #print("dW  fc1 err:",torch.mean(torch.abs(dW_fc1_q/self.scale-dW_fc1)),dW_fc1.max(),dW_fc1.min())

        # reshape to conv5 output shape
        #grad = grad_flat.view(c['pool5'][0].shape)
        grad_q = (grad_flat_q.view(c['pool_q5'][-1].shape)).to(torch.int64)
        grad_q_scaled = False
        self.save_to_cache('grad_pool_q5', grad_q) # big value

        # backward through conv blocks
        for block in reversed(range(1, 6)):
            #pooled_out, indices = c[f'pool{block}']
            first_lid = [1, 2, 3, 5, 7][block - 1]

            pooled_out_q = c[f'pool_q{block}'][-1]
            indices_q = c[f'pool_idx_q{block}'][-1]
            # MaxUnpool to restore pre-pool shape
            #grad = F.max_unpool2d(grad, indices, kernel_size=2, stride=2, output_size=c[f'z{first_lid}'].shape)


            grad_q = F.max_unpool2d(grad_q.to(torch.float64), indices_q, kernel_size=(2,2), stride=(2,2), output_size=c[f'z_q{first_lid}'][-1].shape).to(torch.int64)
            # big value

            # Get conv layer ids in this block
            if block == 1:
                conv_ids = [1]
            elif block == 2:
                conv_ids = [2]
            elif block == 3:
                conv_ids = [3, 4]
            elif block == 4:
                conv_ids = [5, 6]
            elif block == 5:
                conv_ids = [7, 8]


            for lid in reversed(conv_ids):
                if grad_q_scaled:
                    grad_q = grad_q * self.scale
                    grad_q_scaled = False
                self.save_to_cache(f'grad_a_q{lid}', grad_q) # big value
                #grad = grad * (c[f'z{lid}'] > 0)  # ReLU
                grad_q = grad_q * (c[f'a_q{lid}'][-1] > 0)  # ReLU
                
                if not grad_q_scaled:
                    grad_q = torch.round(grad_q.to(torch.float64) / self.scale).to(torch.int64)
                    grad_q_scaled = True
                self.save_to_cache(f'grad_z_q{lid}', grad_q) # small value
                #print("gq Error",torch.abs(grad-grad_q/self.scale).mean(),grad.max(),grad.min())
                # Get input to this conv layer
                if lid == 1:
                #    input_ = self.input
                    input_q = self.input_q
                elif lid == conv_ids[0]:
                #    input_ = c[f'pool{block - 1}'][0] if block > 1 else self.input
                    input_q = c[f'pool_q{block - 1}'][-1] if block > 1 else self.input_q
                else:
                #    input_ = c[f'z{lid - 1}']
                    input_q = c[f'a_q{lid - 1}'][-1]
                
                # Compute weight gradient and update
                #dW = torch.nn.grad.conv2d_weight(input_, W[f'conv{lid}'].shape, grad, padding=1)
                
                dw_Q=torch.nn.grad.conv2d_weight(input_q.to(torch.float64), W[f'conv_q{lid}'].shape, grad_q.to(torch.float64), padding=1)
                dW_q = torch.round(dw_Q/self.scale).to(torch.int64)

                self.save_to_cache(f'dW_conv_q{lid}', dw_Q.to(torch.int64)) # unscaled


                grad_q = F.conv_transpose2d(grad_q.to(torch.float64), W[f'conv_q{lid}'].to(torch.float64), padding=1).to(torch.int64)
                grad_q_scaled = False
                with torch.no_grad():
                    W[f'conv_q{lid}'] -= torch.round(lr * dW_q).to(torch.int64)

            if block == 1:
                self.save_to_cache(f'grad_a_q0', grad_q) # Actually we don't need this
            else:            
                self.save_to_cache(f'grad_pool_q{block - 1}', grad_q)
            
            # if not grad_q_scaled:
            #     grad_q = torch.round(grad_q.to(torch.float64) / self.scale).to(torch.int64)
            #     grad_q_scaled = True

def quantized_softmax(logits_q: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Manually computes softmax with 14-bit fixed-point precision.
    Input: logits_q - float64 tensor, true value
    Output: float64 tensor, quantized value
    """

    # Convert to float for exponentiation
    x = logits_q
    # print(x)

    # For numerical stability: subtract max per row
    x_max = x.max(dim=1, keepdim=True).values
    x_stable = (x - x_max).to(torch.float64) / scale

    # Compute e^x and normalize
    exp_x = torch.exp(x_stable)
    exp_x = torch.round(exp_x*scale).to(torch.int64)
    sum_exp = exp_x.sum(dim=1, keepdim=True)

    softmax = torch.round((exp_x * scale) / sum_exp).to(torch.int64)
    # print(softmax)

    return softmax

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, idx

def train_manual(batch_sz, iter_sz):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # or standard CIFAR-10 mean/std
        ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # batch_sz = 16
    # iter_sz = 8
    loader = DataLoader(dataset , batch_size=batch_sz, shuffle=True)
    subset = Subset(dataset, indices=range(iter_sz * batch_sz))
    indexed_subset = IndexedDataset(subset)  # Wrap to include indices
    loader = DataLoader(indexed_subset, batch_size=batch_sz, shuffle=True)
    model = ManualVGG11()
    lr=0.01
    S=[]

    # Create directory for saving data
    os.makedirs('training_trace/VGG11', exist_ok=True)
    # Save dataset before training
    np.savez(
        'training_trace/VGG11/dataset.npz',
        dataset_inputs=np.stack([
            (subset[i][0] * model.scale).to(torch.int64).numpy()
            for i in range(len(subset))
        ]),
        dataset_labels=np.stack([
            np.eye(10, dtype=np.int64)[subset[i][1]]
            for i in range(len(subset))
        ])
    )
    
    for epoch in range(1):
        correct = total = 0
        correct2= 0
        cnt=0
        epoch_data = []
        epoch_weights = {}

        for x, y, idxs in loader:
            cnt+=1
            # Save input data and labels
            model.save_to_cache('input', (x * model.scale).to(torch.int64))
            model.save_to_cache('label', F.one_hot(y.to(torch.int64), num_classes=10).to(torch.int64))
            model.save_to_cache('index', idxs.to(torch.int64))

            with torch.no_grad():
                z3_q = torch.round(model.forward(x, y) / model.scale).to(torch.int64)
            
            with torch.no_grad():
                probs_q = quantized_softmax(z3_q, model.scale)
                probs_q[range(x.size(0)), y] -= 1 * model.scale   # compute logits, grad
                probs_q = torch.round(probs_q / x.size(0)).to(torch.int64)
                model.save_to_cache('probs_q', probs_q)
            
            with torch.no_grad():
                model.backward_propogation(probs_q, lr=lr)

            correct2 += (z3_q.argmax(1) == y).sum().item()
            total += y.size(0)
            # if cnt%10==0:
            #     print(f" Int Accuracy = {100 * correct2 / total:.2f}%")

        # Convert tensors to numpy arrays before saving
        # cache_np = {}
        # idx = 0
        # for key, value in model.cache.items():
        #     print(key)
        #     print(len(value), value[0].shape)
            # print(value[0][0, 0])
        
        if save_cache_to_file:
            np.savez(f'training_trace/VGG11/epoch_{epoch}.npz', **model.cache)

        # print(epoch_data)

        # print(f"Epoch {epoch+1}: Accuracy = {100 * correct2 / total:.2f}%")
        model.clear_cache()

import pad_conv
if __name__ == "__main__":
    print("VGG11 parameter size:",ManualVGG11().size())
    # torch.manual_seed(0)
    # random.seed(0)
    # train_manual(16, 8)
    # pad_conv.pad_VGG11()
