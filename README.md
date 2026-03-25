# TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation

This repository contains the code for the ICLR 2026 paper **TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation** ([arxiv](https://arxiv.org/abs/2602.04929)).

The current release includes the following features:
  - Implementation of the proposed TurboBoA: `turboboa.py`
  - Quantization of OPT, Llama family models: `main.py`
  - Evaluating the perplexity and 0-shot accuracy (8 tasks) of quantized models

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SamsungLabs/turboboa
   cd turboboa
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The main entry point for quantization and evaluation is `main.py`.

### Basic Usage

To quantize a model (e.g., OPT-125m) with default settings (2-bit weights):

```bash
python main.py --llm_path facebook/opt-125m --w_bits 2 --block_v --n_quant_rows 16 --consider_dX --adaptive_qparam --refine_qparam
```

### TurboBoA Options

- TurboBoA introduces three key features to enhance both the efficiency and accuracy of the conventional [BoA](https://github.com/SamsungLabs/BOA).
  - Feature 1: Joint Quantization of Multiple $N$ Out-channels
  - Feature 2: Error Compensation for Pre-quantized Layers
  - Feature 3: Adaptive Grid Selection with Coordinate Descent-based Refinement

- These features are highly configurable. You can activate or deactivate them by adjusting the following arguments:
   | Feature | Argument | Description |
   | :--- | :--- | :--- |
   | Feature 1 | `--n_quant_rows` | Number of jointly quantized out-channels |
   | Feature 2 | `--consider_dX` | Whether to consider the input deviation induced by the quantization of preceding layers | 
   | | `--alpha` | Percent of the residual term (GPTAQ heuristic) |
   | Feature 3 | `--adaptive_qparam` | Whether to determine qparam adaptively to match the updated distribution |
   | | `--refine_qparam` | Whether to refine qparam after assigning integer weights |
   | | `--n_iters` | Number of iterations for coordinate descent-based scale refinement |

- We also include descriptions of other important arguments for your convenience:
   | Argument | Description |
   | :--- | :--- |
   | `--qparam_comput` | how to select quantization grids. Grids can be determined with a naive MinMax or to minimize the weight perturbation (MMSE) or the layer-wise reconstruction error (Hessian) |
   | `--block_v` | Whether to apply block-wise objective for the value projection. De-activate this option in memory-limited cases (BoA option) |
   | `--act_order_col` | whether to re-order columns before the quantization based on the column-wise Hessian (GPTQ heuristic) |
   | `--act_order_row` | whether to re-order rows before the quantization based on the row-wise Hessian |
    - _**Note)**_ The results reported in the paper represent the best performance achieved by evaluating all four possible combinations of `act_order_col` and `act_order_row` (on/off states).

- For a full list of arguments and their descriptions, please check `utils/process_args.py`.


## License

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/) (CC BY-NC).

## Citation

If you find this work is useful for your research, please cite our paper:

```bibtex
@inproceedings{kimturboboa,
  title={TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation},
  author={Kim, Junhan and Park, Yeo Jeong and Son, Seungwoo and Lee, Chungman and Kim, Ho-young and Kim, Joonyoung and Jeon, Yongkweon},
  booktitle={Fourteenth International Conference on Learning Representations}
}
```
