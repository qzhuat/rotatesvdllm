import cayley_optimize.stiefel_optimizer as stiefel_optimizer
import torch
import numpy as np

def random_orthogonal(dim, A=None):
    """
    Create a orthogonal matrix
    """
    A = np.random.rand(dim, dim)
    Q, _ = np.linalg.qr(A)
    return torch.from_numpy(Q)

def l1_norm_elementwise(W):
    result = 0
    for row in range(W.shape[0]):
        result += torch.norm(W[row,:], p=1)
        
    return result

W = torch.rand(4,4, device='cuda', dtype=torch.float64)
W.requires_grad = False
print('W:', W)

Q = torch.eye(4, dtype=torch.float64)
Q = Q.to(device='cuda')
Q.requires_grad = True

print(torch.norm(Q.T@Q-torch.eye(4, device='cuda')))

optimizer = stiefel_optimizer.SGDG([{'params':[Q],'lr':0.01,'stiefel':True}])

for i in range(100):
    loss = l1_norm_elementwise(Q@W)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
QW = Q @ W
print('QW:', QW)
print(torch.norm(Q.T@Q-torch.eye(4, device='cuda')))

Xs = []
Ys = []
for i in range(10):
    X = torch.rand(4, device='cuda', dtype=torch.float64)
    Y = X @ W
    Xs.append(X)
    Ys.append(Y)

W_metric = torch.abs(W)
W_mask = (torch.zeros_like(W_metric) == 1)
thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*0.5)].cpu()
W_mask = (W_metric<=thresh)

W[W_mask] = 0
print('W after pruning: ', W)


W_metric = torch.abs(QW)
W_mask = (torch.zeros_like(W_metric) == 1)
thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*0.5)].cpu()
W_mask = (W_metric<=thresh)

QW[W_mask] = 0
print('QW after pruning: ', QW)

sum1 = 0
sum2 = 0
for i in range(10):
    X = Xs[i]
    Y = Ys[i]
    loss1 = torch.norm(Y - X@W)
    loss2 = torch.norm(Y - (X@Q.T)@QW)
    
    print(loss1)
    print(loss2)
    
    sum1 += loss1
    sum2 += loss2
    
print(sum1)
print(sum2)