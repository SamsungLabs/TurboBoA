import torch

from utils.quant_utils import damping


def get_cholesky_of_inverse(H):
    U = torch.zeros_like(H)
    for i in range(len(H)):
        compute_cholesky = False
        while not compute_cholesky:
            try:
                U[i] = torch.linalg.cholesky(
                    torch.cholesky_inverse(torch.linalg.cholesky(H[i])), upper=True
                )
                compute_cholesky = True
            except:
                H[i] = damping(H[i])
    
    return U


def reorder_col(W, H_in, dXXT=None):
    org_shape = W.shape
    n_groups, group_size = org_shape[-2], org_shape[-1]
    hidden_size = n_groups * group_size
    W = W.reshape(*org_shape[:-2], hidden_size)
    if H_in.shape[0] == 1:  # Common Hessian for all heads
        W = W.view(1, -1, hidden_size)

    perm = torch.argsort(torch.diagonal(H_in, dim1=-2, dim2=-1), dim=-1, descending=True)
    W = torch.gather(W, dim=-1, index=perm.unsqueeze(-2).expand(-1, W.shape[-2], -1))
    H_in = torch.gather(
        torch.gather(H_in, dim=-2, index=perm.unsqueeze(-1).expand(-1, -1, H_in.shape[-1])),
        dim=-1, index=perm.unsqueeze(-2).expand(-1, H_in.shape[-2], -1)
    )
    if dXXT is not None:
        dXXT = torch.gather(
            torch.gather(dXXT, dim=-2, index=perm.unsqueeze(-1).expand(-1, -1, H_in.shape[-1])),
            dim=-1, index=perm.unsqueeze(-2).expand(-1, H_in.shape[-2], -1)
        )
    invperm = torch.argsort(perm, dim=-1)
  
    W = W.view(*org_shape[:-2], hidden_size).reshape(*org_shape[:-2], n_groups, group_size)

    return W, H_in, dXXT, invperm


def reverse_reorder_col(W, invperm):
    org_shape = W.shape
    n_groups, group_size = org_shape[-2], org_shape[-1]
    hidden_size = n_groups * group_size

    W = W.reshape(*org_shape[:-2], hidden_size).reshape(invperm.shape[0], -1, hidden_size)
    W = torch.gather(W, dim=-1, index=invperm.unsqueeze(-2).expand(-1, W.shape[-2], -1))    
    W = W.reshape(*org_shape[:-2], hidden_size).reshape(*org_shape[:-2], n_groups, group_size)
    
    return W


def reorder_row(W, H_out):
    org_shape = W.shape
    n_groups, group_size = org_shape[-2], org_shape[-1]
    hidden_size = n_groups * group_size
    W = W.reshape(*org_shape[:-2], hidden_size)

    perm = torch.argsort(torch.diagonal(H_out, dim1=-2, dim2=-1), dim=-1, descending=True)
    W = torch.gather(W, dim=-2, index=perm.unsqueeze(-1).expand(-1, -1, W.shape[-1]))
    H_out = torch.gather(
        torch.gather(H_out, dim=-2, index=perm.unsqueeze(-1).expand(-1, -1, H_out.shape[-1])),
        dim=-1, index=perm.unsqueeze(-2).expand(-1, H_out.shape[-2], -1)
    )
    invperm = torch.argsort(perm, dim=-1)

    W = W.reshape(*org_shape[:-2], n_groups, group_size)

    return W, H_out, invperm 
    

def reverse_reorder_row(W, scale, zero, invperm_row):
    org_shape = W.shape
    n_groups, group_size = org_shape[-2], org_shape[-1]
    hidden_size = n_groups * group_size
    W = W.reshape(*org_shape[:-2], hidden_size)
    scale = scale.reshape(*org_shape[:-1])
    zero = zero.reshape(*org_shape[:-1])

    W = torch.gather(W, dim=-2, index=invperm_row.unsqueeze(-1).expand(-1, -1, W.shape[-1]))    
    scale = torch.gather(scale, dim=-2, index=invperm_row.unsqueeze(-1).expand(-1, -1, scale.shape[-1]))    
    zero = torch.gather(zero, dim=-2, index=invperm_row.unsqueeze(-1).expand(-1, -1, zero.shape[-1])) 

    W = W.reshape(*org_shape[:-2], n_groups, group_size)
    scale = scale.reshape(*org_shape[:-1], 1)
    zero = zero.reshape(*org_shape[:-1], 1)
        
    return W, scale, zero
