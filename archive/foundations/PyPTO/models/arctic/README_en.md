# sum_lstm Operator
## Function Description

The sum_lstm operator is the core mechanism of the LSTM-based speculator in the Arctic-Inference framework. It has the following features:

- **Input Fusion**: Weighted fusion of the main state input with additional input signals.
- **RMS Normalization**: Uses RMSNorm instead of LayerNorm to improve computational efficiency.
- **GELU Activation**: Uses an approximate GELU activation function.
- **Gating Mechanism**: Standard LSTM gating logic controlling forget, input, and output.
- **State Update**: Updates the cell state through the forget gate and input gate.
- **Output Generation**: Controls the final hidden state output through the output gate.

This operator is mainly used for speculative decoding in sequence prediction and generation tasks, accelerating the inference process of large language models.

## Mathematical Formulas

The computation process of the Sum LSTM operator is as follows:

1. **Input Fusion**:
   ```
   fused = states_4d + alpha * z4_4d
   ```

2. **Gate Split**:
   Split fused into 4 parts along the last dimension, each of size `D_GATE`:
   ```
   pre_f, pre_i, pre_o, pre_c = split(fused, 4, dim=-1)
   ```

3. **Gate Activation**:
   ```
   f_gate = sigmoid(pre_f)    # Forget gate
   i_gate = sigmoid(pre_i)    # Input gate
   o_gate = sigmoid(pre_o)    # Output gate
   ```

4. **Cell Candidate Processing**:
   ```
   c_cand_norm = RMSNorm(pre_c, eps_cell)
   c_cand_norm = c_cand_norm * w_cell + b_cell  (if weights exist)
   c_act = GELU(c_cand_norm)
   ```

5. **Cell State Update**:
   ```
   c_new = prev_cell * f_gate + c_act * i_gate
   ```

6. **Hidden State Processing**:
   ```
   h_temp = RMSNorm(c_new, eps_state)
   h_temp = h_temp * w_state + b_state  (if weights exist)
   h_act = GELU(h_temp)
   ```

7. **Final Output**:
   ```
   h_new = h_act * o_gate
   ```

Where:
- **RMSNorm(x, eps)**: `x * rsqrt(mean(x^2) + eps)`
- **GELU(x)**: `x * sigmoid(1.702 * x)` (approximate implementation)

## Function Prototype

```python
def sum_lstm_kernel(
    states_4d: pypto.Tensor((BATCH_SIZE, D_GATE_4), pypto.DT_FP16),
    z4_4d: pypto.Tensor((BATCH_SIZE, D_GATE_4), pypto.DT_FP16),
    prev_cell: pypto.Tensor((BATCH_SIZE, D_GATE), pypto.DT_FP16),
    w_cell: pypto.Tensor((D_GATE,), pypto.DT_FP16),
    b_cell: pypto.Tensor((D_GATE,), pypto.DT_FP16),
    w_state: pypto.Tensor((D_GATE,), pypto.DT_FP16),
    b_state: pypto.Tensor((D_GATE,), pypto.DT_FP16),
    config: LstmConfig,
    h_out: pypto.Tensor((BATCH_SIZE, D_GATE), pypto.DT_FP16),
    c_out: pypto.Tensor((BATCH_SIZE, D_GATE), pypto.DT_FP16)
) -> None
```

## Parameter Description

- **states_4d**: Tensor with shape `(BATCH_SIZE, D_GATE_4)` and dtype FP16. Represents the LSTM state input, containing 4 gate signals (forget gate, input gate, output gate, cell candidate).
- **z4_4d**: Tensor with shape `(BATCH_SIZE, D_GATE_4)` and dtype FP16. Represents the additional input signal, fused with states_4d.
- **prev_cell**: Tensor with shape `(BATCH_SIZE, D_GATE)` and dtype FP16. Represents the previous cell state.
- **w_cell**: Tensor with shape `(D_GATE,)` and dtype FP16. Weight parameter for the cell path.
- **b_cell**: Tensor with shape `(D_GATE,)` and dtype FP16. Bias parameter for the cell path.
- **w_state**: Tensor with shape `(D_GATE,)` and dtype FP16. Weight parameter for the state path.
- **b_state**: Tensor with shape `(D_GATE,)` and dtype FP16. Bias parameter for the state path.
- **config**: LstmConfig object containing hyperparameters:
  - `alpha`: Fusion coefficient, default 0.1
  - `eps_cell`: Epsilon for cell RMSNorm, default 1e-6
  - `eps_state`: Epsilon for state RMSNorm, default 1e-6

## Return Value Description

- **h_out**: Tensor with shape `(BATCH_SIZE, D_GATE)` and dtype FP16. Represents the new hidden state output.
- **c_out**: Tensor with shape `(BATCH_SIZE, D_GATE)` and dtype FP16. Represents the new cell state output.

## Call Sample
- For operator source code execution, refer to [test_sum_lstm.py](test_sum_lstm.py).
