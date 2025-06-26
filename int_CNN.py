import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Subset
import torch.nn.functional as F
import numpy as np
import os
import random
import copy

save_cache_to_file = True

class ManualVGG16:
    def __init__(self):
        torch.backends.cudnn.deterministic = True
        self.W = {}

        # Conv layers (no bias), Kaiming init
        conv_shapes = [
            (64, 3, 3, 3), (64, 64, 3, 3),        # Block 1
            (128, 64, 3, 3), (128, 128, 3, 3),    # Block 2
            (256, 128, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3),  # Block 3
            (512, 256, 3, 3), (512, 512, 3, 3), (512, 512, 3, 3),  # Block 4
            (512, 512, 3, 3), (512, 512, 3, 3), (512, 512, 3, 3),  # Block 5
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

    def clear_cache(self):
        self.cache = {}

    def save_to_cache(self, key, value):
        if self.cache.get(key) is None:
            self.cache[key] = []
        self.cache[key].append(copy.deepcopy(value))

    def forward(self, x):
        #self.input = x
        idx = 1
        self.input_q = (x*self.scale).to(torch.int64)
        x_q=self.input_q
        self.save_to_cache('z_q0', x_q)
        
        for block, layers in enumerate([(1, 2), (3, 4), (5, 6, 7), (8, 9, 10), (11, 12, 13)], start=1):
            #for lid in layers:
            #    x = F.conv2d(x, self.W[f'conv{lid}'], padding=1)
            #    self.cache[f'z{lid}'] = x
            #    x = F.relu(x)
            #x, pool_idx = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
            #self.cache[f'pool{block}'] = (x, pool_idx)


            for lid in layers:
                x_q = F.conv2d(x_q.to(torch.float64), self.W[f'conv_q{lid}'].to(torch.float64), padding=1)
                x_q = x_q.to(torch.int64) // self.scale
                x_q = F.relu(x_q)
                self.save_to_cache(f'W_conv_q{lid}', self.W[f'conv_q{lid}'])
                self.save_to_cache(f'z_q{lid}', x_q)

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
        z1_q = torch.round(x_q.to(torch.float64) @ self.W['fc1_q'].to(torch.float64) /self.scale).to(torch.int64)
        a1_q = F.relu(z1_q)
        z2_q = torch.round(a1_q.to(torch.float64) @ self.W['fc2_q'].to(torch.float64) /self.scale).to(torch.int64)
        a2_q = F.relu(z2_q)
        logits_q = torch.round(a2_q.to(torch.float64) @ self.W['fc3_q'].to(torch.float64) /self.scale).to(torch.int64)

        self.save_to_cache('flat_q', x_q)
        self.save_to_cache('W_fc1_q', self.W['fc1_q'])
        self.save_to_cache('z1_q', z1_q)
        self.save_to_cache('a1_q', a1_q)
        self.save_to_cache('W_fc2_q', self.W['fc2_q'])
        self.save_to_cache('z2_q', z2_q)
        self.save_to_cache('a2_q', a2_q)
        self.save_to_cache('W_fc3_q', self.W['fc3_q'])
        self.save_to_cache('logits_q', logits_q)

        return logits_q

    def backward_propogation(self, grad_logits_q,  lr=0.01):
        W = self.W
        c = self.cache
        #print("grad err:",torch.mean(torch.abs(grad_logits-grad_logits_q/self.scale)),grad_logits.max(),grad_logits.min())
        # FC3
        #grad_a2 = grad_logits @ W['fc3'].T
        #dW_fc3 = c['fc2_z'].T @ grad_logits
        #with torch.no_grad():
        #    W['fc3'] -= lr * dW_fc3
        
        grad_a2_q = torch.round(grad_logits_q.to(torch.float64) @ W['fc3_q'].T.to(torch.float64)/self.scale).to(torch.int64)
        dW_fc3_q = torch.round(c['z2_q'][-1].T.to(torch.float64) @ grad_logits_q.to(torch.float64) /self.scale).to(torch.int64)
        #print(dW_fc3_q.shape,dW_fc3.shape,W['fc3_q'].shape)
        with torch.no_grad():
            W['fc3_q'] -= torch.round(lr * dW_fc3_q).to(torch.int64)

        self.save_to_cache('dW_fc3_q', dW_fc3_q)
        self.save_to_cache('grad_a2_q', grad_a2_q)


        # FC2
        #grad_z2 = grad_a2 * (c['fc2_z'] > 0)
        #grad_a1 = grad_z2 @ W['fc2'].T
        #dW_fc2 = c['fc1_z'].T @ grad_z2
        #with torch.no_grad():
        #    W['fc2'] -= lr * dW_fc2
            

        grad_z2_q = grad_a2_q * (c['z2_q'][-1] > 0)
        grad_a1_q = torch.round(grad_z2_q.to(torch.float64) @ W['fc2_q'].T.to(torch.float64) /self.scale).to(torch.int64)
        dW_fc2_q = torch.round(c['z1_q'][-1].T @ grad_z2_q/self.scale).to(torch.int64)
        with torch.no_grad():
            W['fc2_q'] -= torch.round(lr * dW_fc2_q).to(torch.int64)

        self.save_to_cache('grad_z2_q', grad_z2_q)
        self.save_to_cache('dW_fc2_q', dW_fc2_q)
        self.save_to_cache('grad_a1_q', grad_a1_q)

        # FC1
        #grad_z1 = grad_a1 * (c['fc1_z'] > 0)
        #grad_flat = grad_z1 @ W['fc1'].T
        #dW_fc1 = c['flat'].T @ grad_z1
        #with torch.no_grad():
        #    W['fc1'] -= lr * dW_fc1

        grad_z1_q = grad_a1_q * (c['z1_q'][-1] > 0)
        grad_flat_q = torch.round(grad_z1_q @ W['fc1_q'].T /self.scale).to(torch.int64)
        dW_fc1_q = torch.round(c['flat_q'][-1].T @ grad_z1_q/self.scale).to(torch.int64)
        with torch.no_grad():
            W['fc1_q'] -= torch.round(lr * dW_fc1_q).to(torch.int64)

        self.save_to_cache('grad_z1_q', grad_z1_q)
        self.save_to_cache('dW_fc1_q', dW_fc1_q)
        self.save_to_cache('grad_flat_q', grad_flat_q)

        #print("dW  fc1 err:",torch.mean(torch.abs(dW_fc1_q/self.scale-dW_fc1)),dW_fc1.max(),dW_fc1.min())

        # reshape to conv5 output shape
        #grad = grad_flat.view(c['pool5'][0].shape)
        grad_q = grad_flat_q.view(c['pool_q5'][-1].shape)
        self.save_to_cache('grad_pool_q5', grad_q)

        # backward through conv blocks
        for block in reversed(range(1, 6)):
            #pooled_out, indices = c[f'pool{block}']
            first_lid = [1, 3, 5, 8, 11][block - 1]

            pooled_out_q = c[f'pool_q{block}'][-1]
            indices_q = c[f'pool_idx_q{block}'][-1]
            # MaxUnpool to restore pre-pool shape
            #grad = F.max_unpool2d(grad, indices, kernel_size=2, stride=2, output_size=c[f'z{first_lid}'].shape)


            grad_q = F.max_unpool2d(grad_q.to(torch.float64), indices_q, kernel_size=(2,2), stride=(2,2), output_size=c[f'z_q{first_lid}'][-1].shape).to(torch.int64)

            # Get conv layer ids in this block
            if block == 1:
                conv_ids = [1, 2]
            elif block == 2:
                conv_ids = [3, 4]
            elif block == 3:
                conv_ids = [5, 6, 7]
            elif block == 4:
                conv_ids = [8, 9, 10]
            elif block == 5:
                conv_ids = [11, 12, 13]

            for lid in reversed(conv_ids):
                self.save_to_cache(f'grad_z_q{lid}', grad_q)
                #grad = grad * (c[f'z{lid}'] > 0)  # ReLU
                grad_q = grad_q * (c[f'z_q{lid}'][-1] > 0)  # ReLU
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
                    input_q = c[f'z_q{lid - 1}'][-1]
                
                # Compute weight gradient and update
                #dW = torch.nn.grad.conv2d_weight(input_, W[f'conv{lid}'].shape, grad, padding=1)
                dw_Q=torch.nn.grad.conv2d_weight(input_q.to(torch.float64), W[f'conv_q{lid}'].shape, grad_q.to(torch.float64), padding=1)
                dW_q = torch.round(dw_Q/self.scale).to(torch.int64)

                # Propagate grad to previous layer
                #grad = F.conv_transpose2d(grad, W[f'conv{lid}'], padding=1)
                #with torch.no_grad():
                #    W[f'conv{lid}'] -= lr * dW


                grad_q = torch.round(F.conv_transpose2d(grad_q.to(torch.float64), W[f'conv_q{lid}'].to(torch.float64), padding=1)/self.scale).to(torch.int64)
                with torch.no_grad():
                    W[f'conv_q{lid}'] -= torch.round(lr * dW_q).to(torch.int64)
                
                self.save_to_cache(f'dW_conv_q{lid}', dW_q)
            if block == 1:
                self.save_to_cache(f'grad_z_q0', grad_q)
            else:
                self.save_to_cache(f'grad_pool_q{block - 1}', grad_q)
            

def train_manual():
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # or standard CIFAR-10 mean/std
        ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset , batch_size=32, shuffle=True)
    subset = Subset(dataset, indices=range(1024))
    loader = DataLoader(subset, batch_size=32, shuffle=True)
    model = ManualVGG16()
    lr=0.01
    S=[]

    # Create directory for saving data
    os.makedirs('training_trace', exist_ok=True)

    for epoch in range(1):
        correct = total = 0
        correct2= 0
        cnt=0
        epoch_data = []
        epoch_weights = {}

        for x, y in loader:
            cnt+=1
            # Save input data and labels
            epoch_data.append({
                'input': x.numpy(),
                'labels': y.numpy()
            })

            with torch.no_grad():
                logits_q = model.forward(x)
            
            with torch.no_grad():
                probs_q = F.softmax(logits_q/model.scale, dim=1)
                probs_q[range(x.size(0)), y] -= 1   # compute logits, grad
                probs_q /= x.size(0)
                probs_q=torch.round(probs_q*model.scale).to(torch.int64)
                model.save_to_cache('probs_q', probs_q)
            
            with torch.no_grad():
                model.backward_propogation(probs_q, lr=lr)

            correct2 += (logits_q.argmax(1) == y).sum().item()
            total += y.size(0)
            if cnt%10==0:
                print(f" Int Accuracy = {100 * correct2 / total:.2f}%")

        if save_cache_to_file:
            # Convert tensors to numpy arrays before saving
            # cache_np = {}
            # idx = 0
            for key, value in model.cache.items():
                print(key)
                print(len(value), value[0].shape)
            
            np.savez(f'training_trace/epoch_{epoch}.npz', **model.cache)

            # print(epoch_data)

            print(f"Epoch {epoch+1}: Accuracy = {100 * correct2 / total:.2f}%")
            model.clear_cache()

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    train_manual()

