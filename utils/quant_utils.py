import torch


def quantize(x, scale, zero, maxq):
    x_int = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return x_int


def fake_quantize(x, scale, zero, maxq):
    return scale * (quantize(x, scale, zero, maxq) - zero)


def grid_search(w, w_max, w_min, maxq, sym, power, n_grids, H_in, search=True):
    scale = (w_max - w_min) / maxq
    if sym:
        zero = torch.full_like(scale, (maxq + 1) / 2)
    else:
        zero = torch.round(-w_min / scale)

    if not search:
        return scale, zero
    
    else:
        best_score = torch.full_like(w_min, 1e10)
        best_scale = scale
        best_zero = zero

        for i in range(n_grids):
            p = 1 - i / n_grids
            new_w_min = p * w_min
            new_w_max = p * w_max

            for round in ("floor", "ceil"):
                new_scale = (new_w_max - new_w_min) / maxq
                if sym:
                    new_zero = zero
                else:
                    new_zero = torch.floor(-new_w_min / new_scale) if round=="floor" else torch.ceil(-new_w_min / new_scale)
                q = fake_quantize(w, new_scale, new_zero, maxq)

                q -= w
                if H_in is not None:
                    score = torch.sum((q @ H_in) * q, dim=-1, keepdim=True)
                else:
                    q.abs_()            
                    q.pow_(power)
                    score = torch.sum(q, dim=-1, keepdim=True) 
                tmp = score < best_score
                if torch.any(tmp):
                    best_score[tmp] = score[tmp]
                    best_scale[tmp] = new_scale[tmp]
                    best_zero[tmp] = new_zero[tmp]

        return best_scale, best_zero


def optimize_group_qparams(Q, W_org, H_in, dXXT, scale, zero, maxq):
    Q_int = quantize(Q, scale, zero, maxq)
    Q_int_shifted = Q_int - zero
    Q = scale * Q_int_shifted

    org_shape = W_org.shape
    n_groups, group_size = org_shape[-2], org_shape[-1]
    hidden_size = n_groups * group_size
    loss_orig = compute_loss_degradation(W_org, Q, H_in, None, dXXT)

    numerator_temp = torch.einsum('hdni, hnig -> hdng', Q_int_shifted, H_in.reshape(-1, n_groups, group_size, hidden_size))
    if dXXT is not None:
        WdXXTQ = torch.einsum('hdnk, hdk -> hdn', 
            torch.einsum('hdni, hnig -> hdng', Q_int_shifted, dXXT.transpose(-1, -2).reshape(-1, n_groups, group_size, hidden_size)), W_org.reshape(*org_shape[:-2], hidden_size)
        )
    denominator = torch.einsum('hdng, hdng -> hdn',
        torch.einsum('hdni, hnig-> hdng', Q_int_shifted, torch.stack([H_in[:, i*group_size:(i+1)*group_size, i*group_size:(i+1)*group_size] for i in range(n_groups)], dim=1)), Q_int_shifted
    )
    for idx_group in range(n_groups):
        numerator = torch.einsum('hdk, hdk -> hd', numerator_temp[..., idx_group, :], (W_org-Q).reshape(*org_shape[:-2], hidden_size)).unsqueeze(-1)
        if dXXT is not None:
            numerator += WdXXTQ[..., idx_group].unsqueeze(-1)
        scale[..., idx_group, :] += numerator / denominator[..., idx_group].unsqueeze(-1)
        Q = scale * Q_int_shifted
    
    loss_new = compute_loss_degradation(W_org, Q, H_in, None, dXXT)
    print(f"Loss: {loss_orig:.2f} -> {loss_new:.2f}")


    return scale, Q


def compute_loss_degradation(W, Q, H_in, H_row, dXXT):
    if len(W.shape) == 4:
        n_heads, head_dim, n_groups, group_size = W.shape
        d_in = n_groups * group_size

        W = W.reshape(n_heads, head_dim, d_in)
        Q = Q.reshape(n_heads, head_dim, d_in)

    delta_W = Q - W
    if H_row is not None:
        loss = (H_row * (delta_W @ H_in @ delta_W.transpose(-1, -2))).sum()
        if dXXT is not None:
            loss -= 2 * (H_row * (W @ dXXT @ delta_W.transpose(-1, -2))).sum()
    else:    
        loss = ((delta_W @ H_in) * delta_W).sum()    
        if dXXT is not None:
            loss -= 2 * ((W @ dXXT) * delta_W).sum()
    
    return loss


def damping(H, percdamp=.01):  
    # Calculate the mean of diagonals across all heads  
    mean_diags = torch.mean(torch.diagonal(H, dim1=-2, dim2=-1), dim=-1)  

    # Add the damping values back into the original tensor along the diagonals  
    H.diagonal(dim1=-2, dim2=-1).add_(mean_diags.view(-1, *[1]*(len(H.shape)-2)), alpha=percdamp)  

    return H


def filter_dead_neuron(W, H_col, dXXT, replace=1/2048, percdamp=.01, apply_damping=True):
    if len(H_col.shape) == 2:  
        H_col = H_col.unsqueeze(0)  
        if dXXT is not None:
            dXXT = dXXT.unsqueeze(0)
    num_heads, in_features = H_col.shape[0], H_col.shape[-1]  
    W = W.view(num_heads, -1, in_features)  

    # Extract the diagonals of H and find indices where they are equal to 0  
    diagonals = torch.diagonal(H_col, dim1=-2, dim2=-1)  
    idx_dead = (diagonals == 0)  

    # Set the corresponding columns of W to 0 and replace the dead neurons in H with the given value  
    mask = ~idx_dead.unsqueeze(-2)
    W *= mask  
    if dXXT is not None:
        dXXT *= mask
    H_col.diagonal(dim1=-2, dim2=-1)[idx_dead] = replace  

    if apply_damping:
        H_col = damping(H_col, percdamp)

    W = W.view(-1, in_features)  
    H_col = H_col.squeeze()
    if dXXT is not None:
        dXXT = dXXT.squeeze()

    return W, H_col, dXXT
