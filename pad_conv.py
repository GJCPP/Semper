import torch
import torch.nn.functional as F
import math
import numpy as np
import os
# import tqdm

def is_power_of_2(x):
    return (x & (x - 1) == 0) and x > 0

def find_ceiling_log2(x):
    if x == 1:
        return 1
    return math.ceil(math.log2(x))

def pad_weights(X, W, Y, padding_X, pad_right_bottom=True):
    """
    Pad convolution weights and outputs to nearest power-of-2 kernel size,
    supporting multi-sample input.

    Args:
        X: np.ndarray, shape (N, C, n, n)       - input batch of feature maps
        W: np.ndarray, shape (D, C, m, m)       - kernel weights
        Y: np.ndarray, shape (N, D, out, out)   - original outputs
        padding_X: int
        pad_right_bottom: bool (default=True)

    Returns:
        W_pad: np.ndarray, shape (D, C, new_m, new_m)
        Y_pad: np.ndarray, shape (N, D, new_out, new_out)
        new_m: int
        new_padding_X: int
    """
    # Convert to torch
    X = torch.from_numpy(X)
    W = torch.from_numpy(W)
    Y = torch.from_numpy(Y)

    N, C, n, _ = X.shape
    D, Cw, m, _ = W.shape
    assert C == Cw, "Channels mismatch"

    out_sz = n + 2 * padding_X - m + 1
    assert Y.shape == (N, D, out_sz, out_sz)

    # Case: kernel already power of 2
    if is_power_of_2(m):
        return W.clone().numpy(), Y.clone().numpy(), m, padding_X

    # New padded kernel size
    new_m = 1 << find_ceiling_log2(m)
    pad_m = new_m - m
    new_padding_X = padding_X + pad_m

    # Pad weights
    W_pad = torch.zeros((D, C, new_m, new_m), dtype=W.dtype, device=W.device)
    if pad_right_bottom:
        W_pad[:, :, :m, :m] = W
    else:
        W_pad[:, :, pad_m:, pad_m:] = W

    # Recompute Y with padded weights
    # Conv2d expects (N,C,H,W), weight=(D,C,Hk,Wk)
    Y_new = F.conv2d(X, W_pad, padding=new_padding_X)  # (N, D, new_out_sz, new_out_sz)

    return W_pad.numpy(), Y_new.numpy(), new_m, new_padding_X

def flatten_2d(X2d: torch.Tensor, padding: int, new_in: int) -> torch.Tensor:
    """
    Pad a 2D tensor and flatten into 1D of length new_in*new_in.
    """
    n = X2d.shape[0]
    X_pad = torch.zeros((new_in, new_in), dtype=X2d.dtype, device=X2d.device)
    X_pad[padding:padding+n, padding:padding+n] = X2d
    return X_pad.flatten()

def conv2_to_1d(X, W, Y, padding):
    """
    Translate 2D convolution problem into 1D conv form using torch.conv1d,
    supporting batched multi-channel inputs.

    Args:
        X: np.ndarray, shape (N, C, n, n)
        W: np.ndarray, shape (D, C, m, m)     - shared across batch
        Y: np.ndarray, shape (N, D, on, on)
        padding: int

    Returns:
        X_1d: np.ndarray, shape (N, C, L_in)
        Y_1d: np.ndarray, shape (N, D, L_out)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.from_numpy(X).double().to(device)   # (N, C, n, n)
    W = torch.from_numpy(W).double().to(device)   # (D, C, m, m)
    Y = torch.from_numpy(Y).double().to(device)   # (N, D, on, on)

    N, C, n, _ = X.shape
    D, Cw, m, _ = W.shape
    assert C == Cw, "Channel mismatch"

    in_size = n + 2 * padding
    out_sz = n + 2 * padding - m + 1
    assert Y.shape[1:] == (D, out_sz, out_sz), f"Expected (D,{out_sz},{out_sz}), got {Y.shape[1:]}"

    # Input length must be power of 2
    assert is_power_of_2(n), "n must be power of 2"

    # Expand input to nearest power-of-2
    new_in = 1 << find_ceiling_log2(in_size)
    new_padding = (new_in - n) // 2
    new_Y_len = new_in * new_in + m * new_in - 1

    # Flatten inputs (N,C,L)
    X_1d_list = []
    for b in range(N):
        channels = [flatten_2d(X[b, c], new_padding, new_in) for c in range(C)]
        X_1d_list.append(torch.stack(channels, dim=0))  # (C,L)
    X_flat = torch.stack(X_1d_list, dim=0)  # (N,C,L)

    # Build 1D kernels from 2D kernels
    W_1d = torch.zeros((D, C, m, new_in), dtype=W.dtype, device=W.device)
    W_1d[:, :, :m, new_in - m:] = W
    W_1d = W_1d.view(D, C, new_in * m)

    # Apply conv1d across batch
    Y_flat = F.conv1d(X_flat, W_1d, padding=m * new_in - 1)  # (N,D,L_out)

    X_flat_out = torch.round(X_flat).to(torch.int64).cpu().numpy()
    Y_flat_out = torch.round(Y_flat).to(torch.int64).cpu().numpy()
    # print('Y at 1040: ', Y_flat_out[0, 0, 1040])
    return X_flat_out, Y_flat_out

def generate_conv_wit(save_path, save_name, epoch, padding, name_X, name_W, name_Y, X, W, Y, pad_right_bottom):
    B = X.shape[0]
    N = X.shape[1]
    for b in range(B):
        res = {}
        iX = X[b] # [N, C, in, in]
        iW = W[b] # [D, C, k, k]
        iY = Y[b] # [N, D, on, on]
        pW, pY, new_m, new_padding_X = pad_weights(iX, iW, iY, padding_X=padding, pad_right_bottom=pad_right_bottom)
        X_1d, Y_1d = conv2_to_1d(iX, pW, pY, new_padding_X)
        
        pW = np.swapaxes(pW, 0, 1)  # Swap D and C dimensions
        
        res[name_X] = X_1d
        res[name_W] = pW
        res[name_Y] = Y_1d
        
        path = f'{save_path}/epoch_{epoch}_witness/batch_{b}'
        os.makedirs(path, exist_ok=True)
        np.savez(f'{path}/{save_name}', **res)

def pad_VGG16():
    # Path to the .npz file
    model_path = 'training_trace/VGG16'
    file_path = model_path + '/epoch_0.npz'

    # Load the .npz file
    data = np.load(file_path)

    # List all arrays stored in the file
    # print("Arrays in the file:", data.files)

    # Example: Load a specific array (replace 'arr_0' with the actual key if needed)
    array = data['input']


    # Generate witness for forward conv.
    epoch = 0
    for block, layers in enumerate([(1, 2), (3, 4), (5, 6, 7), (8, 9, 10), (11, 12, 13)], start=1):
        print(f'Processing block: {block}/{5}')
        # prove forward
        for layer in layers:
            input_name = f'a_q{layer - 1}' if layer == 1 or layer != layers[0] else f'pool_q{block - 1}'
            X = data[input_name]      # [B, N, C, in, in]
            W = data[f'W_conv_q{layer}'] # [B, D, C, k, k]
            Y = data[f'z_q{layer}']      # [B, N, D, on, on]
            generate_conv_wit(model_path, f'conv_{layer}_forward.npz', epoch, 1,'X', 'W', 'Y', X, W, Y, pad_right_bottom=True)
        # prove backward dW
        for layer in layers:
            input_name = f'a_q{layer - 1}' if layer == 1 or layer != layers[0] else f'pool_q{block - 1}'
            X = data[input_name]      # [B, N, C, in, in]
            dY = data[f'grad_z_q{layer}']      # [B, N, D, on, on]
            dW = data[f'dW_conv_q{layer}'] # [B, D, C, k, k]

            X = np.swapaxes(X, 1, 2) # [B, C, N, in, in]
            dY = np.swapaxes(dY, 1, 2) # [B, D, N, on, on]
            dW = np.swapaxes(dW, 1, 2) # [B, C, D, k, k]

            generate_conv_wit(model_path, f'conv_{layer}_dW.npz', epoch, 1, 'X', 'W', 'Y', X, dY, dW, pad_right_bottom=True)
        # prove backward dX
        for layer in layers:
            input_name = f'grad_a_q{layer - 1}' if layer == 1 or layer != layers[0] else f'grad_pool_q{block - 1}'
            dY = data[f'grad_z_q{layer}'] # [B, N, D, on, on]
            W = data[f'W_conv_q{layer}'] # [B, D, C, k, k]
            dX = data[input_name]      # [B, N, C, in, in]

            W = np.swapaxes(W, 1, 2) # [B, D, C, k, k]
            W = np.flip(W, axis=(3, 4)).copy() # [B, D, C, k, k]

            generate_conv_wit(model_path, f'conv_{layer}_dX.npz', epoch, 1, 'X', 'W', 'Y', dY, W, dX, pad_right_bottom=False)

def pad_VGG11():
    # Path to the .npz file
    model_path = 'training_trace/VGG11'
    file_path = model_path + '/epoch_0.npz'

    # Load the .npz file
    data = np.load(file_path)

    # List all arrays stored in the file
    # print("Arrays in the file:", data.files)

    # Example: Load a specific array (replace 'arr_0' with the actual key if needed)
    array = data['input']


    # Generate witness for forward conv.
    epoch = 0
    for block, layers in enumerate([(1, ), (2, ), (3, 4), (5, 6), (7, 8)], start=1):
        print(f'Processing block: {block}/{5}')
        # prove forward
        for layer in layers:
            input_name = f'a_q{layer - 1}' if layer == 1 or layer != layers[0] else f'pool_q{block - 1}'
            X = data[input_name]      # [B, N, C, in, in]
            W = data[f'W_conv_q{layer}'] # [B, D, C, k, k]
            Y = data[f'z_q{layer}']      # [B, N, D, on, on]
            generate_conv_wit(model_path, f'conv_{layer}_forward.npz', epoch, 1,'X', 'W', 'Y', X, W, Y, pad_right_bottom=True)
        # prove backward dW
        for layer in layers:
            input_name = f'a_q{layer - 1}' if layer == 1 or layer != layers[0] else f'pool_q{block - 1}'
            X = data[input_name]      # [B, N, C, in, in]
            dY = data[f'grad_z_q{layer}']      # [B, N, D, on, on]
            dW = data[f'dW_conv_q{layer}'] # [B, D, C, k, k]

            X = np.swapaxes(X, 1, 2) # [B, C, N, in, in]
            dY = np.swapaxes(dY, 1, 2) # [B, D, N, on, on]
            dW = np.swapaxes(dW, 1, 2) # [B, C, D, k, k]

            generate_conv_wit(model_path, f'conv_{layer}_dW.npz', epoch, 1, 'X', 'W', 'Y', X, dY, dW, pad_right_bottom=True)
        # prove backward dX
        for layer in layers:
            input_name = f'grad_a_q{layer - 1}' if layer == 1 or layer != layers[0] else f'grad_pool_q{block - 1}'
            dY = data[f'grad_z_q{layer}'] # [B, N, D, on, on]
            W = data[f'W_conv_q{layer}'] # [B, D, C, k, k]
            dX = data[input_name]      # [B, N, C, in, in]

            W = np.swapaxes(W, 1, 2) # [B, D, C, k, k]
            W = np.flip(W, axis=(3, 4)).copy() # [B, D, C, k, k]

            generate_conv_wit(model_path, f'conv_{layer}_dX.npz', epoch, 1, 'X', 'W', 'Y', dY, W, dX, pad_right_bottom=False)

def pad_AlexNet():
    # Path to the .npz file
    model_path = 'training_trace/AlexNet'
    file_path = model_path + '/epoch_0.npz'

    # Load the .npz file
    data = np.load(file_path)

    # List all arrays stored in the file
    # print("Arrays in the file:", data.files)

    # Example: Load a specific array (replace 'arr_0' with the actual key if needed)
    array = data['input']


    # Generate witness for forward conv.
    epoch = 0
    for block, layers in enumerate([(1, ), (2, ), (3, ), (4, ), (5, )], start=1):
        print(f'Processing block: {block}/{5}')
        # prove forward
        for layer in layers:
            input_name = f'a_q{layer - 1}' if layer == 1 or layer != layers[0] else f'pool_q{block - 1}'
            X = data[input_name]      # [B, N, C, in, in]
            W = data[f'W_conv_q{layer}'] # [B, D, C, k, k]
            Y = data[f'z_q{layer}']      # [B, N, D, on, on]
            generate_conv_wit(model_path, f'conv_{layer}_forward.npz', epoch, 1,'X', 'W', 'Y', X, W, Y, pad_right_bottom=True)
        # prove backward dW
        for layer in layers:
            input_name = f'a_q{layer - 1}' if layer == 1 or layer != layers[0] else f'pool_q{block - 1}'
            X = data[input_name]      # [B, N, C, in, in]
            dY = data[f'grad_z_q{layer}']      # [B, N, D, on, on]
            dW = data[f'dW_conv_q{layer}'] # [B, D, C, k, k]

            X = np.swapaxes(X, 1, 2) # [B, C, N, in, in]
            dY = np.swapaxes(dY, 1, 2) # [B, D, N, on, on]
            dW = np.swapaxes(dW, 1, 2) # [B, C, D, k, k]

            generate_conv_wit(model_path, f'conv_{layer}_dW.npz', epoch, 1, 'X', 'W', 'Y', X, dY, dW, pad_right_bottom=True)
        # prove backward dX
        for layer in layers:
            input_name = f'grad_a_q{layer - 1}' if layer == 1 or layer != layers[0] else f'grad_pool_q{block - 1}'
            dY = data[f'grad_z_q{layer}'] # [B, N, D, on, on]
            W = data[f'W_conv_q{layer}'] # [B, D, C, k, k]
            dX = data[input_name]      # [B, N, C, in, in]

            W = np.swapaxes(W, 1, 2) # [B, D, C, k, k]
            W = np.flip(W, axis=(3, 4)).copy() # [B, D, C, k, k]

            generate_conv_wit(model_path, f'conv_{layer}_dX.npz', epoch, 1, 'X', 'W', 'Y', dY, W, dX, pad_right_bottom=False)
            
def pad_LeNet():
    # Path to the .npz file
    model_path = 'training_trace/LeNet'
    file_path = model_path + '/epoch_0.npz'

    # Load the .npz file
    data = np.load(file_path)

    # List all arrays stored in the file
    # print("Arrays in the file:", data.files)

    # Example: Load a specific array (replace 'arr_0' with the actual key if needed)
    array = data['input']


    # Generate witness for forward conv.
    epoch = 0
    for block, layers in enumerate([(1, ), (2, )], start=1):
        print(f'Processing block: {block}/{2}')
        # prove forward
        for layer in layers:
            input_name = f'a_q{layer - 1}' if layer == 1 or layer != layers[0] else f'pool_q{block - 1}'
            X = data[input_name]      # [B, N, C, in, in]
            W = data[f'W_conv_q{layer}'] # [B, D, C, k, k]
            Y = data[f'z_q{layer}']      # [B, N, D, on, on]
            generate_conv_wit(model_path, f'conv_{layer}_forward.npz', epoch, 1,'X', 'W', 'Y', X, W, Y, pad_right_bottom=True)
        # prove backward dW
        for layer in layers:
            input_name = f'a_q{layer - 1}' if layer == 1 or layer != layers[0] else f'pool_q{block - 1}'
            X = data[input_name]      # [B, N, C, in, in]
            dY = data[f'grad_z_q{layer}']      # [B, N, D, on, on]
            dW = data[f'dW_conv_q{layer}'] # [B, D, C, k, k]

            X = np.swapaxes(X, 1, 2) # [B, C, N, in, in]
            dY = np.swapaxes(dY, 1, 2) # [B, D, N, on, on]
            dW = np.swapaxes(dW, 1, 2) # [B, C, D, k, k]

            generate_conv_wit(model_path, f'conv_{layer}_dW.npz', epoch, 1, 'X', 'W', 'Y', X, dY, dW, pad_right_bottom=True)
        # prove backward dX
        for layer in layers:
            input_name = f'grad_a_q{layer - 1}' if layer == 1 or layer != layers[0] else f'grad_pool_q{block - 1}'
            dY = data[f'grad_z_q{layer}'] # [B, N, D, on, on]
            W = data[f'W_conv_q{layer}'] # [B, D, C, k, k]
            dX = data[input_name]      # [B, N, C, in, in]

            W = np.swapaxes(W, 1, 2) # [B, D, C, k, k]
            W = np.flip(W, axis=(3, 4)).copy() # [B, D, C, k, k]

            generate_conv_wit(model_path, f'conv_{layer}_dX.npz', epoch, 1, 'X', 'W', 'Y', dY, W, dX, pad_right_bottom=False)

if __name__ == "__main__":
    pad_AlexNet()