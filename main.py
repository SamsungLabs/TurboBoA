import io
import time
from contextlib import redirect_stdout

from quantize import turboboa_fwrd
from utils.data_utils import get_calib_data
from utils.eval_utils import evaluate
from utils.model_utils import get_model
from utils.process_args import get_turboboa_arguments, get_turboboa_weight_quant_infos

if __name__ == '__main__':
    args = get_turboboa_arguments()
    
    # load model
    with redirect_stdout(io.StringIO()) as f:
        llm = get_model(args.llm_path)
    llm.seqlen = args.seqlen
    llm.eval()

    # evaluate the fp model performance
    if args.eval_fp:
        results = evaluate(llm, args)
        print(results)
        exit(0)

    # load calib. data
    calib_data = get_calib_data(args)

    # quantize
    qconfigs, turboboa_opts, hyperparams = get_turboboa_weight_quant_infos(args)
    print("Start quantization")
    tick = time.time()
    turboboa_fwrd(llm, calib_data, qconfigs, turboboa_opts, hyperparams, args)
    process_time = round(time.time() - tick, 3)
    print(f"Quantization processing time: {process_time}")
    
    # evaluate
    print(args)
    results = evaluate(llm, args)
    results['time'] = process_time
    print(results)
