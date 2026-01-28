import functools

import torch

from quantizers.minmax import MinMaxQuantizer
from quantizers.turboboa import TurboBoA
from utils.model_utils import cache_first_transformer_input, get_head_info, get_rotary_emb, get_transformer_blocks
from utils.utils import cleanup_memory, find_layers

QKV_NAMES = {"query": "self_attn.q_proj", "key": "self_attn.k_proj", "value": "self_attn.v_proj"}


@torch.no_grad()
def turboboa_fwrd(llm, calib_data, qconfigs, turboboa_opts: dict, hyperparams: dict, args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cache = llm.config.use_cache
    llm.config.use_cache = False

    # cache inputs for fast quantization
    quant_inps, block_kwargs = cache_first_transformer_input(llm, calib_data)
    fp_inps = quant_inps.clone() if turboboa_opts["consider_dX"] else None

    transformer_blocks = get_transformer_blocks(llm)
    n_heads, n_kv_heads, head_dim = get_head_info(llm)
    rotary_emb = get_rotary_emb(llm)
    rotary_matrix = get_rotary_matrix(rotary_emb, llm.config, block_kwargs['position_ids'].cpu()) if rotary_emb is not None else None
    
    # quantize each Transformer block
    ngpus = torch.cuda.device_count()
    for i in range(len(transformer_blocks)):
        print(f'>>>> Quantizing {i+1}-th Transformer Block.... ({i+1}/{len(transformer_blocks)})')
        transformer_block = transformer_blocks[i].to(dev)
        
        fp_layers = find_layers(transformer_block)

        wrappers = {}
        for name, fp_layer in fp_layers.items():
            wrappers[name] = TurboBoA(fp_layer, turboboa_opts, hyperparams)
            wrappers[name].quantizer = MinMaxQuantizer()
            wrappers[name].quantizer.configure(qconfigs["w_bits"], per_channel=True, group_size=qconfigs["group_size"], sym=qconfigs["w_sym"], mse=False)
            wrappers[name].quantizer.find_params(wrappers[name].layer.weight.data)

        # compute Hessians
        block_v = turboboa_opts['block_v']
        if ngpus > 1:
            fp_inps = compute_Hessian_multigpu(transformer_block, n_heads, n_kv_heads, head_dim, wrappers, quant_inps, fp_inps, block_kwargs, block_v, rotary_matrix)
        else:
            fp_inps = compute_Hessian(transformer_block, n_heads, n_kv_heads, head_dim, wrappers, quant_inps, fp_inps, block_kwargs, block_v, rotary_matrix)

        # quantize
        for name in fp_layers:
            print('-' * 50)
            print(f">>> Layer: {name}")
            wrappers[name].quant(args.print_memory_usage)
            wrappers[name].free()

        # cache inputs for next transformer block
        for j in range(len(quant_inps)):
            quant_inps[j] = transformer_block(quant_inps[j].unsqueeze(0), **block_kwargs)[0]
        
        transformer_blocks[i] = transformer_block.cpu()
        del transformer_block
        del wrappers 
        
        cleanup_memory(verbose=False)

    llm.config.use_cache = use_cache


def get_rotary_matrix(rotary_emb, config, position_ids):
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    half_head_dim = head_dim // 2
    seqlen = position_ids.shape[-1]

    cos, sin = rotary_emb(torch.rand([1], dtype=torch.float32), position_ids=position_ids)
    cos, sin = cos.squeeze(), sin.squeeze()

    rotary_matrix = torch.zeros(*(seqlen, head_dim, head_dim), dtype=cos.dtype, device=cos.device)
    rotary_matrix[:, :half_head_dim, :half_head_dim] = torch.diag_embed(cos[:, :half_head_dim])
    rotary_matrix[:, :half_head_dim, half_head_dim:] = -torch.diag_embed(sin[:, :half_head_dim])
    rotary_matrix[:, half_head_dim:, :half_head_dim] = torch.diag_embed(sin[:, :half_head_dim])
    rotary_matrix[:, half_head_dim:, half_head_dim:] = torch.diag_embed(cos[:, :half_head_dim])

    rotary_matrix = rotary_matrix.unsqueeze(dim=1)
    return rotary_matrix


@torch.no_grad()
def compute_Hessian(transformer_block, n_heads, n_kv_heads, head_dim, wrappers, quant_inps, fp_inps, block_kwargs, block_v, rotary_matrix):
    from utils.hessian_utils import CovarianceCollector, compute_cov, preprocess
    consider_dX = fp_inps is not None

    layers = find_layers(transformer_block)
    cov_collectors = {}
    for name, layer in layers.items():
        cov_collectors[name] = CovarianceCollector(layer)
    if rotary_matrix is not None:  # For models exploiting RoPE, we need to save the covariance of outputs after RoPE.
        cov_collectors['rot_out_Q'] = CovarianceCollector(transformer_block.self_attn.rot_out_Q)
        cov_collectors['rot_out_K'] = CovarianceCollector(transformer_block.self_attn.rot_out_K)

    handles = []
    for name in layers:
        if name in [QKV_NAMES["query"], QKV_NAMES["key"]]:  # Q, K, V share inputs.
            pass
        elif name == QKV_NAMES["value"]:
            handles.append(layers[name].register_forward_hook(cov_collectors[name].compute_cov_in_batch))
            if block_v or consider_dX:
                handles.append(layers[name].register_forward_hook(cov_collectors[name].save_inps))  # we need to compute XATAXT for value
            if consider_dX:
                handles.append(layers[name].register_forward_hook(cov_collectors[name].compute_cov_res_in_batch))
        else:
            handles.append(layers[name].register_forward_hook(cov_collectors[name].compute_cov_in_batch))
            if consider_dX:
                handles.append(layers[name].register_forward_hook(cov_collectors[name].save_inps))
                handles.append(layers[name].register_forward_hook(cov_collectors[name].compute_cov_res_in_batch))
    if rotary_matrix is None:
        handles.append(layers[QKV_NAMES["query"]].register_forward_hook(functools.partial(cov_collectors[QKV_NAMES["query"]].compute_cov_out_batch, n_heads=n_heads)))
        handles.append(layers[QKV_NAMES["key"]].register_forward_hook(functools.partial(cov_collectors[QKV_NAMES["key"]].compute_cov_out_batch, n_heads=n_kv_heads)))
    else:
        handles.append(transformer_block.self_attn.rot_out_Q.register_forward_hook(functools.partial(cov_collectors['rot_out_Q'].compute_cov_out_batch, n_heads=n_heads)))
        handles.append(transformer_block.self_attn.rot_out_K.register_forward_hook(functools.partial(cov_collectors['rot_out_K'].compute_cov_out_batch, n_heads=n_kv_heads)))

    if block_v:
        block_kwargs = block_kwargs.copy()
        block_kwargs['output_attentions'] = True
        XXT_value, n_data_in_value = 0, 0
        if consider_dX:
            dXXT_value, n_data_res_in_value = 0, 0
    
    cache_fp_list = [True, False] if consider_dX else [False]
    for j in range(len(quant_inps)):
        for cache_fp in cache_fp_list:
            for cov_collector in cov_collectors.values():
                cov_collector.cache_fp = cache_fp
            
            if cache_fp:
                outs = transformer_block(fp_inps[j].unsqueeze(0), **block_kwargs)
                fp_inps[j] = outs[0]
                if block_v:
                    fp_inp_value = cov_collectors[QKV_NAMES["value"]].fp_inp[0]
                    fp_A = outs[-1]
                    fp_inp_value = torch.einsum('bhli, bid -> bhld', fp_A, fp_inp_value)
                    fp_inp_value = preprocess(fp_inp_value, n_heads)
            else:
                if not block_v:
                    transformer_block(quant_inps[j].unsqueeze(0), **block_kwargs)
                else:
                    quant_A = transformer_block(quant_inps[j].unsqueeze(0), **block_kwargs)[-1]
                    quant_inp_value = cov_collectors[QKV_NAMES["value"]].quant_inp.pop()
                    quant_inp_value = torch.einsum('bhli, bid -> bhld', quant_A, quant_inp_value)
                    quant_inp_value = preprocess(quant_inp_value, n_heads)
                    
                    XXT_value, n_data_in_value = compute_cov(XXT_value, n_data_in_value, quant_inp_value)
                    if consider_dX:
                        res = fp_inp_value - quant_inp_value
                        dXXT_value, n_data_res_in_value = compute_cov(dXXT_value, n_data_res_in_value, res, quant_inp_value)
            
    for h in handles:
        h.remove()
    
    # Assign H_in except for value
    for name, wrapper in wrappers.items():
        if name in [QKV_NAMES["query"], QKV_NAMES["key"]]:
            wrapper.H_in = cov_collectors[QKV_NAMES["value"]].XXT
            if consider_dX:
                wrapper.dXXT = cov_collectors[QKV_NAMES["value"]].dXXT
        elif name == QKV_NAMES["value"]:
            pass
        else:
            wrapper.H_in = cov_collectors[name].XXT
            if consider_dX:
                wrapper.dXXT = cov_collectors[name].dXXT

    # Assign H_in for value
    if block_v:
        if n_kv_heads != n_heads:
            n_shared = n_heads // n_kv_heads
            hidden_size = XXT_value.shape[-1]
            XXT_value = XXT_value.reshape(n_kv_heads, n_shared, hidden_size, hidden_size).mean(dim=1)
            if consider_dX:
                dXXT_value = dXXT_value.reshape(n_kv_heads, n_shared, hidden_size, hidden_size).mean(dim=1)
        
        wrappers[QKV_NAMES['value']].H_in = XXT_value
        del XXT_value
        if consider_dX:
            wrappers[QKV_NAMES['value']].dXXT = dXXT_value
            del dXXT_value

    else:
        wrappers[QKV_NAMES['value']].H_in = cov_collectors[QKV_NAMES["value"]].XXT
        if consider_dX:
            wrappers[QKV_NAMES['value']].dXXT = cov_collectors[QKV_NAMES["value"]].dXXT

    # Assign H_out for query/key
    if n_kv_heads != n_heads:
        cov_collectors['rot_out_Q'].YYT = cov_collectors['rot_out_Q'].YYT.reshape(n_kv_heads, n_shared, head_dim, head_dim).mean(dim=1)
        cov_collectors['rot_out_K'].YYT = cov_collectors['rot_out_K'].YYT[:, None, :, :].expand(-1, n_shared, -1, -1).reshape(n_heads, head_dim, head_dim)
    
    if rotary_matrix is None:
        wrappers[QKV_NAMES["query"]].H_out = cov_collectors[QKV_NAMES["key"]].YYT
        wrappers[QKV_NAMES["key"]].H_out = cov_collectors[QKV_NAMES["query"]].YYT
    else:
        rotary_matrix = rotary_matrix.cuda()
        wrappers[QKV_NAMES["query"]].H_out = (rotary_matrix.transpose(-1, -2) @ cov_collectors['rot_out_K'].YYT @ rotary_matrix).mean(0)
        wrappers[QKV_NAMES["key"]].H_out = (rotary_matrix.transpose(-1, -2) @ cov_collectors['rot_out_Q'].YYT @ rotary_matrix).mean(0)

    del cov_collectors
    cleanup_memory(verbose=False)

    return fp_inps


@torch.no_grad()
def compute_Hessian_multigpu(transformer_block, n_heads, n_kv_heads, head_dim, wrappers, quant_inps, fp_inps, block_kwargs, block_v, rotary_matrix):
    gpus = [torch.device("cuda:%d" % i) for i in range(torch.cuda.device_count())]
    from utils.hessian_utils import CovarianceCollector, compute_cov, preprocess
    consider_dX = fp_inps is not None

    layers = find_layers(transformer_block)
    cov_collectors = {}
    for name, layer in layers.items():
        cov_collectors[name] = CovarianceCollector(layer)
    if rotary_matrix is not None:  # For models exploiting RoPE, we need to save the covariance of outputs after RoPE.
        cov_collectors['rot_out_Q'] = CovarianceCollector(transformer_block.self_attn.rot_out_Q)
        cov_collectors['rot_out_K'] = CovarianceCollector(transformer_block.self_attn.rot_out_K)

    handles = []
    for name in layers:
        if name in [QKV_NAMES["query"], QKV_NAMES["key"]]:  # Q, K, V share inputs.
            pass
        elif name == QKV_NAMES["value"]:
            handles.append(layers[name].register_forward_hook(cov_collectors[name].compute_cov_in_batch))
            if block_v or consider_dX:
                handles.append(layers[name].register_forward_hook(cov_collectors[name].save_inps))  # we need to compute XATAXT for value
            if consider_dX:
                handles.append(layers[name].register_forward_hook(cov_collectors[name].compute_cov_res_in_batch))
        else:
            handles.append(layers[name].register_forward_hook(cov_collectors[name].compute_cov_in_batch))
            if consider_dX:
                handles.append(layers[name].register_forward_hook(cov_collectors[name].save_inps))
                handles.append(layers[name].register_forward_hook(cov_collectors[name].compute_cov_res_in_batch))
    if rotary_matrix is None:
        handles.append(layers[QKV_NAMES["query"]].register_forward_hook(functools.partial(cov_collectors[QKV_NAMES["query"]].compute_cov_out_batch, n_heads=n_heads)))
        handles.append(layers[QKV_NAMES["key"]].register_forward_hook(functools.partial(cov_collectors[QKV_NAMES["key"]].compute_cov_out_batch, n_heads=n_kv_heads)))
    else:
        handles.append(transformer_block.self_attn.rot_out_Q.register_forward_hook(functools.partial(cov_collectors['rot_out_Q'].compute_cov_out_batch, n_heads=n_heads)))
        handles.append(transformer_block.self_attn.rot_out_K.register_forward_hook(functools.partial(cov_collectors['rot_out_K'].compute_cov_out_batch, n_heads=n_kv_heads)))

    if block_v:
        block_kwargs = block_kwargs.copy()
        block_kwargs['output_attentions'] = True
        XXT_value, n_data_in_value = 0, 0
        if consider_dX:
            dXXT_value, n_data_res_in_value = 0, 0
    
    cache_fp_list = [True, False] if consider_dX else [False]
    for j in range(len(quant_inps)):
        for cache_fp in cache_fp_list:
            for cov_collector in cov_collectors.values():
                cov_collector.cache_fp = cache_fp
            
            if cache_fp:
                outs = transformer_block(fp_inps[j].unsqueeze(0), **block_kwargs)
                fp_inps[j] = outs[0]
                if block_v:
                    fp_inp_value = cov_collectors[QKV_NAMES["value"]].fp_inp[0]
                    fp_A = outs[-1]
                    fp_inp_value = torch.einsum('bhli, bid -> bhld', fp_A, fp_inp_value).to(device=gpus[-1])
                    fp_inp_value = preprocess(fp_inp_value, n_heads)
            else:
                if not block_v:
                    transformer_block(quant_inps[j].unsqueeze(0), **block_kwargs)
                else:
                    quant_A = transformer_block(quant_inps[j].unsqueeze(0), **block_kwargs)[-1]
                    quant_inp_value = cov_collectors[QKV_NAMES["value"]].quant_inp.pop()
                    quant_inp_value = torch.einsum('bhli, bid -> bhld', quant_A, quant_inp_value).to(device=gpus[-1])
                    quant_inp_value = preprocess(quant_inp_value, n_heads)
                    
                    XXT_value, n_data_in_value = compute_cov(XXT_value, n_data_in_value, quant_inp_value)
                    if consider_dX:
                        res = fp_inp_value - quant_inp_value
                        dXXT_value, n_data_res_in_value = compute_cov(dXXT_value, n_data_res_in_value, res, quant_inp_value)
            
    for h in handles:
        h.remove()
    
    # Assign H_in except for value
    for name, wrapper in wrappers.items():
        if name in [QKV_NAMES["query"], QKV_NAMES["key"]]:
            wrapper.H_in = cov_collectors[QKV_NAMES["value"]].XXT
            if consider_dX:
                wrapper.dXXT = cov_collectors[QKV_NAMES["value"]].dXXT
        elif name == QKV_NAMES["value"]:
            pass
        else:
            wrapper.H_in = cov_collectors[name].XXT
            if consider_dX:
                wrapper.dXXT = cov_collectors[name].dXXT

    # Assign H_in for value
    if block_v:
        if n_kv_heads != n_heads:
            n_shared = n_heads // n_kv_heads
            hidden_size = XXT_value.shape[-1]
            XXT_value = XXT_value.reshape(n_kv_heads, n_shared, hidden_size, hidden_size).mean(dim=1)
            if consider_dX:
                dXXT_value = dXXT_value.reshape(n_kv_heads, n_shared, hidden_size, hidden_size).mean(dim=1)
        
        wrappers[QKV_NAMES['value']].H_in = XXT_value
        del XXT_value
        if consider_dX:
            wrappers[QKV_NAMES['value']].dXXT = dXXT_value
            del dXXT_value

    else:
        wrappers[QKV_NAMES['value']].H_in = cov_collectors[QKV_NAMES["value"]].XXT
        if consider_dX:
            wrappers[QKV_NAMES['value']].dXXT = cov_collectors[QKV_NAMES["value"]].dXXT

    # Assign H_out for query/key
    if n_kv_heads != n_heads:
        n_shared = n_heads // n_kv_heads
        cov_collectors['rot_out_Q'].YYT = cov_collectors['rot_out_Q'].YYT.reshape(n_kv_heads, n_shared, head_dim, head_dim).mean(dim=1)
        cov_collectors['rot_out_K'].YYT = cov_collectors['rot_out_K'].YYT[:, None, :, :].expand(-1, n_shared, -1, -1).reshape(n_heads, head_dim, head_dim)
    
    if rotary_matrix is None:
        wrappers[QKV_NAMES["query"]].H_out = cov_collectors[QKV_NAMES["key"]].YYT
        wrappers[QKV_NAMES["key"]].H_out = cov_collectors[QKV_NAMES["query"]].YYT
    else:
        rotary_matrix = rotary_matrix.cuda()
        wrappers[QKV_NAMES["query"]].H_out = (rotary_matrix.transpose(-1, -2) @ cov_collectors['rot_out_K'].YYT @ rotary_matrix).mean(0)
        wrappers[QKV_NAMES["key"]].H_out = (rotary_matrix.transpose(-1, -2) @ cov_collectors['rot_out_Q'].YYT @ rotary_matrix).mean(0)

    del cov_collectors
    cleanup_memory(verbose=False)

    return fp_inps
