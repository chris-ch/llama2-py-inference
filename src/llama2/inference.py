import itertools
import logging
import sys
import time
import math
import struct
from typing import List, BinaryIO, Tuple, TextIO
from dataclasses import dataclass, field

import numpy
from numpy.typing import NDArray


@dataclass
class TransformerWeighting:
    token_embedding_table: NDArray = None
    rms_att_weight: NDArray = None
    wq: NDArray = None
    wk: NDArray = None
    wv: NDArray = None
    wo: NDArray = None
    rms_ffn_weight: NDArray = None
    w1: NDArray = None
    w3: NDArray = None
    w2: NDArray = None
    rms_final_weight: NDArray = None
    freq_cis_real: NDArray[NDArray[numpy.float32]] = None
    freq_cis_imag: NDArray[NDArray[numpy.float32]] = None


@dataclass
class Network:
    dim: int
    hidden_dim: int
    n_layers: int
    num_attention_heads: int  # number of attention heads for each attention layer in the Transformer encoder
    num_key_value_heads: int  # number of key_value heads that should be used to implement Grouped Query Attention
    vocab_size: int  # number of different tokens that can be represented by the inputs passed when calling LlamaModel
    seq_len: int  # maximum sequence length that this model might ever be used with
    weighting: TransformerWeighting = None
    head_dimension: int = field(init=False)

    def __post_init__(self):
        self.head_dimension = self.dim // self.num_attention_heads


@dataclass
class RunState:
    scores: NDArray[NDArray[numpy.float32]] = None  # scores/attention values (n_heads, seq_len)
    key_cache: NDArray[NDArray[NDArray[numpy.float32]]] = None  # (layer, seq_len, dim)
    value_cache: NDArray[NDArray[NDArray[numpy.float32]]] = None  # (layer, seq_len, dim)


def checkpoint_init_weights(conf: Network, file: BinaryIO) -> TransformerWeighting:
    def read_as_array(count: int) -> NDArray[numpy.float32]:
        return numpy.array(struct.unpack(f'{count}f', file.read(count * 4)), dtype=numpy.float32)

    def read_as_array2(nrows: int, ncols: int) -> NDArray[NDArray[numpy.float32]]:
        return numpy.array(struct.unpack(f'{nrows * ncols}f', file.read(nrows * ncols * 4)),
                           dtype=numpy.float32).reshape((nrows, ncols))

    def read_as_array3(ndepth: int, nrows: int, ncols: int) -> NDArray[NDArray[NDArray[numpy.float32]]]:
        return numpy.array(struct.unpack(f'{nrows * ncols * ndepth}f', file.read(ndepth * nrows * ncols * 4)),
                           dtype=numpy.float32).reshape(
            (ndepth, ncols, nrows))

    weights = TransformerWeighting()
    weights.token_embedding_table = read_as_array2(conf.vocab_size, conf.dim)
    weights.rms_att_weight = read_as_array2(conf.n_layers, conf.dim)
    weights.wq = read_as_array3(conf.n_layers, conf.dim, conf.dim)
    weights.wk = read_as_array3(conf.n_layers, conf.dim, conf.dim)
    weights.wv = read_as_array3(conf.n_layers, conf.dim, conf.dim)
    weights.wo = read_as_array3(conf.n_layers, conf.dim, conf.dim)
    weights.rms_ffn_weight = read_as_array2(conf.n_layers, conf.dim)
    weights.w1 = read_as_array3(conf.n_layers, conf.dim, conf.hidden_dim)
    weights.w2 = read_as_array3(conf.n_layers, conf.hidden_dim, conf.dim)
    weights.w3 = read_as_array3(conf.n_layers, conf.dim, conf.hidden_dim)
    weights.rms_final_weight = read_as_array(conf.dim)
    weights.freq_cis_real = read_as_array2(conf.seq_len, conf.head_dimension // 2)
    weights.freq_cis_imag = read_as_array2(conf.seq_len, conf.head_dimension // 2)
    return weights


def tokenizer_init(file: BinaryIO, size: int) -> Tuple[List[str], List[float]]:
    vocab, vocab_scores = [], []
    _ = struct.unpack('i', file.read(4))[0]
    for _ in range(size):
        vocab_scores.append(struct.unpack('f', file.read(4))[0])
        length = struct.unpack('i', file.read(4))[0]
        bstr = file.read(length)
        if type(bstr) is not str:
            bstr = bstr.decode('utf8')
        vocab.append(bstr)
    return vocab, vocab_scores


def rms_norm(x: NDArray, weight: NDArray) -> NDArray:
    # calculate sum of squares
    ss = numpy.divide(numpy.sum(numpy.power(x, 2)), len(x)) + 1e-5
    # normalize and scale
    return numpy.multiply(numpy.multiply(weight, x), 1.0 / numpy.sqrt(ss))


def softmax(values: NDArray, size: int) -> NDArray:
    max_val = numpy.max(values[:size])
    exp_values = numpy.exp(values[:size] - max_val)
    softmax_values = exp_values / numpy.sum(exp_values)
    return numpy.concatenate((softmax_values, values[size:]))


def transformer(token_code: int, step_count: int, network: Network, state: RunState) -> NDArray[numpy.float32]:
    # getting the token embedding
    token = network.weighting.token_embedding_table[token_code]

    # plucking out the current row of freq_cis_real and freq_cis_imag
    freq_cis_real_row: NDArray[numpy.float32] = network.weighting.freq_cis_real[step_count]
    freq_cis_imag_row: NDArray[numpy.float32] = network.weighting.freq_cis_imag[step_count]

    # forwarding all the layers
    for index_layer in range(network.n_layers):
        # Attention rmsnorm
        residual_branch_activation = rms_norm(token, network.weighting.rms_att_weight[index_layer])

        # QKV matmuls for this position
        w_q = network.weighting.wq[index_layer]
        heads_q = apply_rotations(network, freq_cis_real_row, freq_cis_imag_row, residual_branch_activation, w_q)

        w_k = network.weighting.wk[index_layer]
        heads_k = apply_rotations(network, freq_cis_real_row, freq_cis_imag_row, residual_branch_activation, w_k)
        state.key_cache[step_count, index_layer] = heads_k

        w_v = network.weighting.wv[index_layer]
        heads_v = numpy.dot(w_v, residual_branch_activation).reshape(network.num_attention_heads, network.head_dimension)
        state.value_cache[step_count, index_layer] = heads_v

        # Multihead attention. Iterate over all heads
        for index_head in range(network.num_attention_heads):
            # Iterate over all timesteps, including the current one
            for timestep in range(step_count + 1):
                # Get the key vector for this head and at this timestep
                key_vector = state.key_cache[timestep, index_layer, index_head]

                # Calculate the attention score as the dot product of q and k
                score = numpy.divide(numpy.dot(heads_q[index_head], key_vector), math.sqrt(network.head_dimension))

                # Save the score to the attention buffer
                state.scores[index_head, timestep] = score

            # Softmax the scores to get attention weights, from 0..pos inclusively
            state.scores[index_head] = softmax(state.scores[index_head], step_count + 1)

            # Weighted sum of the values, store back into residual branch activation
            head_activation = residual_branch_activation[
                              index_head * network.head_dimension:(index_head + 1) * network.head_dimension]
            head_activation[:] = numpy.zeros(network.head_dimension)
            for timestep in range(step_count + 1):
                value_vector = state.value_cache[timestep, index_layer, index_head]
                attention_weight: numpy.float32 = state.scores[index_head, timestep]
                head_activation[:] += numpy.multiply(value_vector, attention_weight)

        # Final matrix multiplication to get the output of the attention and residual branch activation back into token
        token = numpy.add(token, numpy.dot(network.weighting.wo[index_layer], residual_branch_activation))

        # Feed-forward Neural Network
        residual_branch_activation = rms_norm(token, network.weighting.rms_ffn_weight[index_layer])

        hidden_dimension_buffer1 = numpy.dot(network.weighting.w1[index_layer], residual_branch_activation)
        hidden_dimension_buffer2 = numpy.dot(network.weighting.w3[index_layer], residual_branch_activation)

        sigmoid_linear_unit = numpy.vectorize(lambda value: value / (1. + math.exp(-value)))
        hidden_dimension_buffer1 = numpy.multiply(sigmoid_linear_unit(hidden_dimension_buffer1),
                                                  hidden_dimension_buffer2)
        residual_branch_activation = numpy.dot(network.weighting.w2[index_layer], hidden_dimension_buffer1)

        # Residual connection
        token = numpy.add(token, residual_branch_activation)

    # Final rmsnorm
    token = rms_norm(token, network.weighting.rms_final_weight)

    # Classifier into logits
    return numpy.dot(network.weighting.token_embedding_table, token)


def apply_rotations(network, freq_cis_real_row: NDArray[numpy.float32],
                    freq_cis_imag_row: NDArray[numpy.float32],
                    residual_branch_activation: NDArray[numpy.float32],
                    weights: NDArray[NDArray[numpy.float32]]) -> NDArray[NDArray[numpy.float32]]:
    heads_values = numpy.dot(weights, residual_branch_activation).reshape(network.num_attention_heads, network.head_dimension)
    # Apply RoPE rotation to the q and k vectors for each head
    for index_head, head_item_index in itertools.product(range(network.num_attention_heads),
                                                         range(0, network.head_dimension, 2)):
        head, head_next = perform_rope_rotation(
            freq_cis_real_row[head_item_index // 2],
            freq_cis_imag_row[head_item_index // 2],
            heads_values[index_head][head_item_index],
            heads_values[index_head][head_item_index + 1]
        )
        heads_values[index_head][head_item_index] = head
        heads_values[index_head][head_item_index + 1] = head_next
    return heads_values


def perform_rope_rotation(freq_cis_real: numpy.float32,
                          freq_cis_imag: numpy.float32,
                          head_value: numpy.float32,
                          head_value_next: numpy.float32,
                          ) -> Tuple[numpy.float32, numpy.float32]:
    updated_head_q = head_value * freq_cis_real - head_value_next * freq_cis_imag
    updated_head_q_next = head_value * freq_cis_imag + head_value_next * freq_cis_real
    return updated_head_q, updated_head_q_next


def bpe_encode(prompt: str, vocab: List[str], vocab_scores: NDArray[numpy.float32]) -> List[int]:
    tokens = []
    # First encode every individual character in the input text
    for char in prompt:
        pos = vocab.index(char)
        tokens.append(pos)

    # Merge the best consecutive pair each iteration, according to the scores in vocab_scores
    return _process_tokens(tokens, vocab, vocab_scores)


def _process_tokens(tokens: List[int], vocab: List[str], vocab_scores: NDArray[numpy.float32]) -> List[int]:
    while True:
        best_score = -1e10
        best_id = -1
        best_idx = -1

        for count, token_pair in enumerate(zip(tokens[:-1], tokens[1:])):
            token_prev, token_next = token_pair
            # Check if we can merge the pair (tokens[i], tokens[i+1])
            merged_tokens = vocab[token_prev] + vocab[token_next]
            try:
                pos = vocab.index(merged_tokens)
            except ValueError:
                pos = -1
            if pos != -1 and vocab_scores[pos] > best_score:
                # This merge pair exists in vocab! Record its score and position
                best_score = vocab_scores[pos]
                best_id = pos
                best_idx = count

        if best_idx == -1:
            break  # We couldn't find any more pairs to merge, so we're done

        # Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id
        del tokens[best_idx + 1]

    return tokens


def time_in_ms() -> float:
    # Returns time in milliseconds for benchmarking the model speed
    return int(time.time() * 1000)


def draw_sample(probabilities: NDArray) -> int:
    r = numpy.random.random()
    cdf = numpy.cumsum(probabilities)
    return numpy.argmax(r < cdf)


def run(model_file: BinaryIO, tokenizer_file: BinaryIO, temperature: float, max_steps: int, prompt: str, seed: int,
        output: TextIO=sys.stdout):
    if seed is None:
        seed = int(time.time())
    numpy.random.seed(seed)

    # Read in the model.bin file
    network = _load_network(model_file)

    vocab, vocab_scores = tokenizer_init(tokenizer_file, network.vocab_size)
    # Create and initialize the application RunState
    state = _make_init_state(network)

    prompt_tokens: List[int] = bpe_encode(prompt, vocab,
                                          numpy.array(vocab_scores, dtype=numpy.float32)) if prompt else numpy.array([],
                                                                                                                     dtype=numpy.float32)
    # Start the main loop
    start: float = 0.  # Used to time our code, only initialized after the first iteration
    # Initialize with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    token_code: int = 1
    timestep: int = 0  # Position in the sequence
    # Explicitly print the initial BOS token for stylistic symmetry reasons
    print("<s>", flush=True, file=output)

    # Right now we cannot run for more than config.seq_len steps
    if max_steps <= 0 or max_steps > network.seq_len:
        max_steps = network.seq_len

    start = time_in_ms()
    result = generate_tokens(network, state, max_steps, prompt_tokens, temperature, vocab, output)

    # Report achieved tok/s
    end = time_in_ms()
    logging.info(f"achieved tok/s: {(max_steps - 1) / (end - start) * 1000:.01f}")


def _load_network(file: BinaryIO) -> Network:
    # Read in the config header
    network_config_file = file.read(struct.calcsize('7i'))
    # Unpacking the data
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack('7i', network_config_file)
    network = Network(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
    logging.info(f"created network: {network}")
    network.vocab_size = abs(network.vocab_size)
    network.weighting = checkpoint_init_weights(network, file)
    return network


def _make_init_state(network: Network) -> RunState:
    state = RunState()
    state.scores = numpy.zeros(shape=(network.num_attention_heads, network.seq_len))
    state.key_cache = numpy.zeros(
        shape=(network.seq_len, network.n_layers, network.num_attention_heads, network.head_dimension))
    state.value_cache = numpy.zeros(
        shape=(network.seq_len, network.n_layers, network.num_attention_heads, network.head_dimension))
    return state


def generate_tokens(network: Network, state: RunState, checked_max_steps: int, prompt_tokens: List[int], temperature: float, vocab: List[str], output) -> List[str]:
    result = []
    token_code: int = 1
    timestep: int = 0  # Position in the sequence
    while timestep < checked_max_steps:

        token_str, next_token = generate_next_token(timestep, prompt_tokens, temperature, network, vocab, token_code, state)
        result.append(token_str)

        print(token_str, end="", flush=True, file=output)

        if next_token == 1:
            break

        # Advance forward
        token_code = next_token
        timestep += 1

    return result


def generate_next_token(timestep, prompt_tokens, temperature, network, vocab, token_code, state) -> Tuple[str, int]:
    # Forward the transformer to get logits for the next token
    logits = transformer(token_code, timestep, network, state)

    if timestep < len(prompt_tokens):
        # If we are still processing the input prompt, force the next prompt token
        next_token = prompt_tokens[timestep]
    elif temperature == 0.0:
        # Greedy argmax sampling: take the token with the highest probability
        next_token = numpy.argmax(logits)
    else:
        # Apply the temperature to the logits
        # Apply softmax to the logits to get the probabilities for the next token
        # Sample from this distribution to get the next token
        next_token = draw_sample(softmax(numpy.divide(logits, temperature), network.vocab_size))

    # Following BOS token (1), sentencepiece decoder strips any leading whitespace
    token_str = (
        vocab[next_token].lstrip()
        if token_code == 1 and vocab[next_token][0] == ' ' else vocab[next_token]
    )
    return token_str, next_token
