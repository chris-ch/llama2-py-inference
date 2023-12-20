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
    token_embedding_table: NDArray[NDArray[numpy.float32]] = None
    rms_att_weight: List[NDArray[numpy.float32]] = None
    wq: List[NDArray[NDArray[numpy.float32]]] = None
    wk: List[NDArray[NDArray[numpy.float32]]] = None
    wv: List[NDArray[NDArray[numpy.float32]]] = None
    wo: List[NDArray[NDArray[numpy.float32]]] = None
    rms_ffn_weight: List[NDArray[numpy.float32]] = None
    w1: List[NDArray[NDArray[numpy.float32]]] = None
    w2: List[NDArray[NDArray[numpy.float32]]] = None
    w3: List[NDArray[NDArray[numpy.float32]]] = None
    rms_final_weight: NDArray[numpy.float32] = None
    freq_cis_real: List[NDArray[numpy.float32]] = None
    freq_cis_imag: List[NDArray[numpy.float32]] = None


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
class RunCache:
    key_cache: List[List[List[NDArray[numpy.float32]]]] = field(default_factory=list)
    value_cache: List[List[List[NDArray[numpy.float32]]]] = field(default_factory=list)


def checkpoint_init_weights(conf: Network, file: BinaryIO) -> TransformerWeighting:
    def read_as_array(count: int) -> NDArray[numpy.float32]:
        return numpy.array(struct.unpack(f'{count}f', file.read(count * 4)), dtype=numpy.float32)

    def read_as_matrix(nrows: int, ncols: int) -> NDArray[NDArray[numpy.float32]]:
        return numpy.array(struct.unpack(f'{nrows * ncols}f', file.read(nrows * ncols * 4)),
                           dtype=numpy.float32).reshape((nrows, ncols))

    def read_as_vector_list(nrows: int, ncols: int) -> List[NDArray[numpy.float32]]:
        values = struct.unpack(f'{nrows * ncols}f', file.read(nrows * ncols * 4))
        return [numpy.array(values[i:i + ncols], dtype=numpy.float32) for i in range(0, len(values), ncols)]

    def read_as_matrix_list(ndepth: int, nrows: int, ncols: int) -> List[NDArray[NDArray[numpy.float32]]]:
        values = struct.unpack(f'{nrows * ncols * ndepth}f', file.read(ndepth * nrows * ncols * 4))
        return [numpy.array(values[i:i + nrows * ncols], dtype=numpy.float32).reshape((ncols, nrows)) for i in range(0, len(values), nrows * ncols)]

    weights = TransformerWeighting()
    weights.token_embedding_table = read_as_matrix(conf.vocab_size, conf.dim)
    weights.rms_att_weight = read_as_vector_list(conf.n_layers, conf.dim)
    weights.wq = read_as_matrix_list(conf.n_layers, conf.dim, conf.dim)
    weights.wk = read_as_matrix_list(conf.n_layers, conf.dim, conf.dim)
    weights.wv = read_as_matrix_list(conf.n_layers, conf.dim, conf.dim)
    weights.wo = read_as_matrix_list(conf.n_layers, conf.dim, conf.dim)
    weights.rms_ffn_weight = read_as_vector_list(conf.n_layers, conf.dim)
    weights.w1 = read_as_matrix_list(conf.n_layers, conf.dim, conf.hidden_dim)
    weights.w2 = read_as_matrix_list(conf.n_layers, conf.hidden_dim, conf.dim)
    weights.w3 = read_as_matrix_list(conf.n_layers, conf.dim, conf.hidden_dim)
    weights.rms_final_weight = read_as_array(conf.dim)
    weights.freq_cis_real = read_as_vector_list(conf.seq_len, conf.head_dimension // 2)
    weights.freq_cis_imag = read_as_vector_list(conf.seq_len, conf.head_dimension // 2)
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


def rms_norm(x: NDArray[numpy.float32], weight: NDArray[numpy.float32]) -> NDArray[numpy.float32]:
    # calculate sum of squares
    ss: numpy.float32 = numpy.divide(numpy.sum(numpy.power(x, 2)), len(x)) + 1e-5
    # normalize and scale
    return numpy.multiply(numpy.multiply(weight, x), 1.0 / numpy.sqrt(ss))


def softmax(values: NDArray, size: int) -> NDArray:
    max_val = numpy.max(values[:size])
    exp_values = numpy.exp(values[:size] - max_val)
    softmax_values = exp_values / numpy.sum(exp_values)
    return numpy.concatenate((softmax_values, values[size:]))


def transformer(token_code: int, step_count: int, network: Network, cache: RunCache) -> NDArray[numpy.float32]:
    # getting the token embedding
    token: NDArray[numpy.float32] = network.weighting.token_embedding_table[token_code]

    # plucking out the current row of freq_cis_real and freq_cis_imag
    freq_cis_real_row: NDArray[numpy.float32] = network.weighting.freq_cis_real[step_count]
    freq_cis_imag_row: NDArray[numpy.float32] = network.weighting.freq_cis_imag[step_count]

    # forwarding all the layers
    for index_layer in range(network.n_layers):
        heads_q, heads_k, heads_v = compute_qkv(network, index_layer, freq_cis_real_row, freq_cis_imag_row, token)
        cache.key_cache[step_count].append(heads_k)
        cache.value_cache[step_count].append(heads_v)
        activations: NDArray[NDArray[numpy.float32]] = multihead_activation(network, index_layer, cache, heads_q)
        delta_token_qkv = numpy.dot(network.weighting.wo[index_layer], activations.reshape(network.dim))

        # Final matrix multiplication to get the output of the attention and residual branch activation back into token
        token = numpy.add(token, delta_token_qkv)

        # Feed-forward Neural Network
        delta_token_ffn = compute_delta_ffn(network, index_layer, token)

        # Residual connection
        token = numpy.add(token, delta_token_ffn)

    # Final rmsnorm
    token = rms_norm(token, network.weighting.rms_final_weight)

    # Classifier into logits
    return numpy.dot(network.weighting.token_embedding_table, token)


def compute_delta_ffn(network: Network, index_layer: int, token: NDArray[numpy.float32]) -> NDArray[numpy.float32]:
    rba = rms_norm(token, network.weighting.rms_ffn_weight[index_layer])
    hidden_dimension_buffer1 = numpy.dot(network.weighting.w1[index_layer], rba)
    hidden_dimension_buffer2 = numpy.dot(network.weighting.w3[index_layer], rba)
    sigmoid_linear_unit = numpy.vectorize(lambda value: value / (1. + math.exp(-value)))
    hidden_dimension_buffer1 = numpy.multiply(sigmoid_linear_unit(hidden_dimension_buffer1),
                                              hidden_dimension_buffer2)
    return numpy.dot(network.weighting.w2[index_layer], hidden_dimension_buffer1)


def compute_qkv(network: Network,
                index_layer: int,
                freq_cis_real_row: NDArray[numpy.float32],
                freq_cis_imag_row: NDArray[numpy.float32],
                token: NDArray[numpy.float32]) -> Tuple[List[NDArray[numpy.float32]], List[NDArray[numpy.float32]], List[NDArray[numpy.float32]]]:
    # Attention rmsnorm
    rba: NDArray[numpy.float32] = rms_norm(token, network.weighting.rms_att_weight[index_layer])
    # QKV matmuls for this position
    w_q: NDArray[NDArray[numpy.float32]] = numpy.dot(network.weighting.wq[index_layer], rba).reshape(
        network.num_attention_heads,
        network.head_dimension)
    heads_q: List[NDArray[numpy.float32]] = [apply_rotations(network, head, freq_cis_real_row, freq_cis_imag_row) for
                                             head in w_q.tolist()]
    w_k = numpy.dot(network.weighting.wk[index_layer], rba).reshape(network.num_attention_heads,
                                                                     network.head_dimension)
    heads_k: List[NDArray[numpy.float32]] = [apply_rotations(network, head, freq_cis_real_row, freq_cis_imag_row) for
                                             head in w_k.tolist()]
    w_v: NDArray[NDArray[numpy.float32]] = network.weighting.wv[index_layer]
    heads_v: List[NDArray[numpy.float32]] = list(
        numpy.dot(w_v, rba).reshape(
            network.num_attention_heads, network.head_dimension
        )
    )
    return heads_q, heads_k, heads_v


def multihead_activation(network: Network, index_layer: int, cache: RunCache, heads_q: List[NDArray[numpy.float32]]) -> NDArray[NDArray[numpy.float32]]:
    activations: List[NDArray[numpy.float32]] = []
    # Multihead attention. Iterate over all heads
    for index_head in range(network.num_attention_heads):
        raw_scores: List[numpy.float32] = compute_scores(network.head_dimension, cache, index_layer, index_head, heads_q)
        # Softmax the scores to get attention weights, from 0..pos inclusively
        head_scores: NDArray[numpy.float32] = softmax(numpy.array(raw_scores), len(raw_scores)).tolist()
        # Weighted sum of the values, store back into residual branch activation
        current_activation: NDArray[numpy.float32] = build_activation(network.head_dimension, index_layer, cache.value_cache, index_head, head_scores)
        activations.append(current_activation)
    return numpy.array(activations)


def build_activation(dimension: int, index_layer: int, value_cache: List[List[List[NDArray[numpy.float32]]]], index_head: int, head_scores: List[numpy.float32]) -> NDArray[numpy.float32]:
    current_activation = numpy.zeros(shape=dimension)
    for count in range(len(value_cache)):
        value_vector: NDArray[numpy.float32] = value_cache[count][index_layer][index_head]
        attention_weight: numpy.float32 = head_scores[count]
        current_activation += numpy.multiply(value_vector, attention_weight)
    return current_activation


def compute_scores(head_dimension: int, cache: RunCache, index_layer: int, index_head: int, heads_q: List[NDArray[numpy.float32]]) -> List[numpy.float32]:
    head_scores: List[numpy.float32] = []
    # Iterate over all timesteps, including the current one
    for key_cache in cache.key_cache:
        # Get the key vector for this head and at this timestep
        key_vector: NDArray[numpy.float32] = key_cache[index_layer][index_head]
        # Calculate the attention score as the dot product of q and k
        score = numpy.divide(numpy.dot(heads_q[index_head], key_vector), math.sqrt(head_dimension))
        # Save the score to the attention buffer
        head_scores.append(score)
    return head_scores


def apply_rotations(network: Network, head: List[numpy.float32], freq_cis_real_row: NDArray[numpy.float32],
                    freq_cis_imag_row: NDArray[numpy.float32]) -> NDArray[numpy.float32]:
    result = []
    for head_item_index in range(0, network.head_dimension, 2):
        real = freq_cis_real_row[head_item_index // 2]
        imag = freq_cis_imag_row[head_item_index // 2]
        value = head[head_item_index]
        value_next = head[head_item_index + 1]
        result.append(value * real - value_next * imag)
        result.append(value * imag + value_next * real)
    return numpy.array(result)


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
        output: TextIO = sys.stdout):
    if seed is None:
        seed = int(time.time())
    numpy.random.seed(seed)

    # Read in the model.bin file
    network = _load_network(model_file)

    vocab, vocab_scores = tokenizer_init(tokenizer_file, network.vocab_size)

    prompt_tokens: List[int] = bpe_encode(prompt, vocab,
                                          numpy.array(vocab_scores, dtype=numpy.float32)) if prompt else numpy.array([],
                                                                                                                     dtype=numpy.float32)
    # Explicitly print the initial BOS token for stylistic symmetry reasons
    print("<s>", flush=True, file=output)

    # Right now we cannot run for more than config.seq_len steps
    if max_steps <= 0 or max_steps > network.seq_len:
        max_steps = network.seq_len

    start = time_in_ms()
    result = generate_tokens(network, max_steps, prompt_tokens, temperature, vocab, output)

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


def generate_tokens(network: Network, checked_max_steps: int, prompt_tokens: List[int], temperature: float, vocab: List[str], output) -> List[str]:
    result = []
    token_code: int = 1
    timestep: int = 0  # Position in the sequence

    # Create and initialize the application RunState
    cache = RunCache()
    cache.key_cache = []
    cache.value_cache = []

    while timestep < checked_max_steps:
        cache.key_cache.append([])
        cache.value_cache.append([])
        token_str, next_token = generate_next_token(timestep, prompt_tokens, temperature, network, vocab, token_code, cache)
        result.append(token_str)

        print(token_str, end="", flush=True, file=output)

        if next_token == 1:
            break

        # Advance forward
        token_code = next_token
        timestep += 1

    return result


def generate_next_token(timestep: int, prompt_tokens: List[int], temperature: float, network: Network, vocab: List[str],
                        token_code: int, cache: RunCache) -> Tuple[str, int]:
    # Forward the transformer to get logits for the next token
    logits = transformer(token_code, timestep, network, cache)

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
