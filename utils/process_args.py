import argparse
from pathlib import Path
import transformers

def get_turboboa_arguments(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument("--cache_dir", type=str, default='cache')
    parser.add_argument("--print_memory_usage", action='store_true')
    
    ## Model
    parser.add_argument("--llm_path", type=str, default='facebook/opt-125m')
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--eval_fp", action='store_true', help='Whether to evaluate the original fp model performance')
    
    ## Calib. Data
    parser.add_argument('--calib_data', type=str, default="wikitext2", choices=["c4", "wikitext2", "redpajama"])
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--seqlen', type=int, default=2048, help='Length of input sequences')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')

    ## Quant. Configs.
    parser.add_argument('--w_bits', type=int, default=2)
    parser.add_argument('--group_size', type=int, default=-1)
    parser.add_argument('--w_sym', action="store_true")
    
    ## TurboBoA Options
    parser.add_argument('--qparam_comput', type=str, default='Hessian', choices=['MinMax', 'MMSE', 'Hessian'], help="How to determine Quant. Params")
    parser.add_argument('--block_v', action="store_true", help="Whether to apply block-wise objective for the value projection. De-activate this option in memory-limited cases")
    parser.add_argument('--n_quant_rows', type=int, default=16, help="Number of jointly quantized out-channels (feature 1).")
    parser.add_argument('--consider_dX', action='store_true', help='Whether to consider the input deviation induced by the quantization of preceding layers (feature 2).')
    parser.add_argument('--alpha', type=float, default=.25, help='Percent of the residual term (feature 2). GPTAQ heuristic.')
    parser.add_argument('--adaptive_qparam', action='store_true', help='Whether to determine qparam adaptively to match the updated distribution (feature 3).')
    parser.add_argument('--refine_qparam', action='store_true', help='Whether to refine qparam after assigning integer weights (feature 3).')
    parser.add_argument('--n_iters', type=int, default=1, help='Number of iterations for coordinate descent-based scale refinement (feature 3).')
    parser.add_argument('--act_order_col', action='store_true')
    parser.add_argument('--act_order_row', action='store_true')

    parser.add_argument('--replace', type=float, default=1, help='Value to be replaced for the Hessian diagonal elements corresponding to dead neurons')
    
    # LM Eval Arguments
    parser.add_argument("--lm_eval", action="store_true", help="Evaluate the model on LM Eval tasks.")
    parser.add_argument('--tasks', nargs='+', default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada_openai", "lambada_standard", "openbookqa", "boolq"])
    parser.add_argument('--lm_eval_batch_size', default="auto", help='Batch size for evaluating with lm eval harness.')
    
    args = parser.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    if args.tokenizer_path is None:
        args.tokenizer_path = args.llm_path
    args.llm_name = args.tokenizer_path.split('/')[-1]
    args.llm_type = args.llm_name.split('-')[0]
    args.tokenizer_cls = transformers.AutoTokenizer

    if args.group_size != -1:
        args.act_order_col = False

    args.replace = 1 / args.seqlen

    return args


def get_turboboa_weight_quant_infos(args):
    qconfigs = {
        "w_bits": args.w_bits,
        "group_size": args.group_size,
        "w_sym": args.w_sym
    }
    turboboa_opts = {
        "qparam_comput": args.qparam_comput,
        "block_v": args.block_v,
        "n_quant_rows": args.n_quant_rows,
        "adaptive_qparam": args.adaptive_qparam,
        "refine_qparam": args.refine_qparam,
        "n_iters": args.n_iters,
        "consider_dX": args.consider_dX,
        "alpha": args.alpha,
        'act_order_col': args.act_order_col, 
        'act_order_row': args.act_order_row, 
    }
    hyperparams = {"replace": args.replace}
    
    return qconfigs, turboboa_opts, hyperparams


def set_qmodel_path(args, qmodel_dir="qmodels"):
    qmodel_dir += "/boa" if args.n_quant_rows == 1 else "/turboboa"
    qmodel_name = f"{args.llm_name}-w{args.w_bits}g{args.group_size}-{args.calib_data}-{args.qparam_comput.lower()}-n_quant_rows_{args.n_quant_rows}"
    if args.group_size == -1:
        qmodel_name = qmodel_name.replace("g-1", "")
    if args.consider_dX:
        qmodel_name += f"-F2-alpha_{args.alpha}"
    if not args.adaptive_qparam:
        qmodel_name += "-fixed_qparam"
    if args.refine_qparam:
        qmodel_name += "-F3"
    if args.act_order_col:
        qmodel_name += "-col"
    if args.act_order_row:
        qmodel_name += "-row"

    return f"{qmodel_dir}/{qmodel_name}"