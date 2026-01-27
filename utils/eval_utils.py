import os
import torch
from torch import nn

from .model_utils import get_transformer_blocks, cache_first_transformer_input, set_device_after_last_transformer_block, get_logits_from_last_hidden_states
from .data_utils import get_testdata
from .utils import cleanup_memory

@torch.no_grad()
def evaluate(llm, args):
    results = {}

    # ppl evaluation
    print('ppl performance')
    ppl_results = {}
    for dataset in ["wikitext2", 'c4-new']:
        cache_testloader = f'{args.cache_dir}/testloader_{args.llm_type}_{dataset}_{args.seqlen}.cache'

        if os.path.exists(cache_testloader):
            testloader = torch.load(cache_testloader)
            print(f"load calibration from {cache_testloader}")
        else:
            testloader = get_testdata(dataset, args)
            torch.save(testloader, cache_testloader)
        
        ppl = eval_ppl(llm, testloader)
        print(f'{dataset} : {ppl :.3f}')
        ppl_results[dataset] = round(ppl, 3)

    # zero-shot evaluation
    print('zero-shot performance')
    zero_shot_results = eval_zero_shot(llm, args) if args.lm_eval else {}

    results = {**ppl_results, **zero_shot_results}

    return results


def eval_ppl(llm, testloader):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seqlen = llm.seqlen
    use_cache = llm.config.use_cache
    llm.config.use_cache = False
    
    transformer_blocks = get_transformer_blocks(llm)
    inps, block_kwargs = cache_first_transformer_input(llm, testloader)

    for i in range(len(transformer_blocks)):
        transformer_block = transformer_blocks[i].to(dev)
        for j in range(len(testloader)):
            inps[j] = transformer_block(inps[j].unsqueeze(0), **block_kwargs)[0]
        transformer_blocks[i] = transformer_block.cpu()
        
        del transformer_block
        cleanup_memory(verbose=False)

    set_device_after_last_transformer_block(llm, dev)

    nlls = []
    for i in range(len(testloader)):
        lm_logits = get_logits_from_last_hidden_states(llm, inps[i].unsqueeze(0))
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testloader[i][0][:, 1:].to(dev)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (len(testloader) * seqlen))
    llm.config.use_cache = use_cache

    return ppl.item()


@torch.no_grad()
def eval_zero_shot(llm, args):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    if torch.cuda.device_count() > 1:
        from accelerate.utils import get_balanced_memory
        from accelerate import infer_auto_device_map, dispatch_model

        no_split_module_classes = ["LlamaDecoderLayer"]
        max_memory = get_balanced_memory(llm, no_split_module_classes=no_split_module_classes)
        device_map = infer_auto_device_map(llm, max_memory=max_memory, no_split_module_classes=no_split_module_classes)
        dispatch_model(llm, device_map=device_map, offload_buffers=True, offload_dir="offload", state_dict=llm.state_dict())
        cleanup_memory(verbose=False)

    else:
        llm.cuda()
    tokenizer = args.tokenizer_cls.from_pretrained(args.tokenizer_path)

    zero_shot_results = {}
    fail_some_tasks = False
    for task in args.tasks:
        try:
            hflm = HFLM(pretrained=llm, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size, trust_remote_code=True)
            result = simple_evaluate(
                hflm,
                tasks=[task],
                num_fewshot=0,
                batch_size=args.lm_eval_batch_size,
            )
            zero_shot_results[task] = round(100 * result['results'][task].get('acc_norm,none', result['results'][task]['acc,none']), 2)
        except:
            fail_some_tasks = True
    
    if fail_some_tasks:
        from utils.process_args import set_qmodel_path
        qmodel_path = set_qmodel_path(args)
        llm.save_pretrained(qmodel_path)
    else:
        zero_shot_results['lambada'] = (zero_shot_results.pop('lambada_openai') + zero_shot_results.pop("lambada_standard")) / 2
        zero_shot_results['acc_avg'] = round(sum(zero_shot_results.values()) / len(zero_shot_results.values()), 2)
    print(zero_shot_results)
    llm.cpu()

    return zero_shot_results