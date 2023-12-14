import logging
import sys
import time
import math
import struct
from typing import List, BinaryIO, Tuple
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
    freq_cis_real: NDArray[numpy.float64] = None
    freq_cis_imag: NDArray[numpy.float64] = None
    wcls: NDArray = None


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
    x: NDArray = None
    xb: NDArray = None
    q: NDArray = None
    k: NDArray = None
    v: NDArray = None
    att: NDArray = None
    key_cache: NDArray = None
    value_cache: NDArray = None
    xb2: NDArray = None
    hb: NDArray = None
    hb2: NDArray = None
    logits: NDArray = None


def checkpoint_init_weights(conf: Network, file: BinaryIO) -> TransformerWeighting:
    def read_floats(count: int) -> NDArray:
        return numpy.array(struct.unpack(f'{count}f', file.read(count * 4)))

    def read_floats_as_array2(nrows: int, ncols: int) -> NDArray:
        return numpy.array(struct.unpack(f'{nrows * ncols}f', file.read(nrows * ncols * 4))).reshape((nrows, ncols))

    def read_floats_as_array3(ndepth: int, nrows: int, ncols: int) -> NDArray:
        return numpy.array(struct.unpack(f'{nrows * ncols * ndepth}f', file.read(ndepth * nrows * ncols * 4))).reshape((ndepth, ncols, nrows))

    weights = TransformerWeighting()
    weights.token_embedding_table = read_floats_as_array2(conf.vocab_size, conf.dim)
    weights.rms_att_weight = read_floats_as_array2(conf.n_layers, conf.dim)
    weights.wq = read_floats_as_array3(conf.n_layers, conf.dim, conf.dim)
    weights.wk = read_floats_as_array3(conf.n_layers, conf.dim, conf.dim)
    weights.wv = read_floats_as_array3(conf.n_layers, conf.dim, conf.dim)
    weights.wo = read_floats_as_array3(conf.n_layers, conf.dim, conf.dim)
    weights.rms_ffn_weight = read_floats_as_array2(conf.n_layers, conf.dim)
    weights.w1 = read_floats_as_array3(conf.n_layers, conf.dim, conf.hidden_dim)
    weights.w2 = read_floats_as_array3(conf.n_layers, conf.hidden_dim, conf.dim)
    weights.w3 = read_floats_as_array3(conf.n_layers, conf.dim, conf.hidden_dim)
    weights.rms_final_weight = read_floats(conf.dim)
    weights.freq_cis_real = read_floats_as_array2(conf.seq_len, (conf.dim // conf.num_attention_heads) // 2)
    weights.freq_cis_imag = read_floats_as_array2(conf.seq_len, (conf.dim // conf.num_attention_heads) // 2)
    weights.wcls = weights.token_embedding_table
    return weights


def tokenizer_init(file: BinaryIO, size: int) -> Tuple[List[str], List[float], int]:
    vocab, vocab_scores = [], []
    max_token_length = struct.unpack('i', file.read(4))[0]
    for _ in range(size):
        vocab_scores.append(struct.unpack('f', file.read(4))[0])
        length = struct.unpack('i', file.read(4))[0]
        bstr = file.read(length)
        if type(bstr) is not str:
            bstr = bstr.decode('utf8')
        vocab.append(bstr)
    return vocab, vocab_scores, max_token_length


def rmsnorm(x: NDArray, weight: NDArray) -> NDArray:
    # calculate sum of squares
    ss = numpy.divide(numpy.sum(numpy.power(x, 2)), len(x)) + 1e-5
    # normalize and scale
    return numpy.multiply(numpy.multiply(weight, x), 1.0 / numpy.sqrt(ss))


def softmax(values: NDArray, size: int) -> NDArray:
    max_val = numpy.max(values[:size])
    exp_values = numpy.concatenate((numpy.exp(values[:size] - max_val), values[size:]))
    return numpy.divide(exp_values, numpy.sum(exp_values))


# token, pos, config, state, weights
def transformer(token: int, step_count: int, network: Network, state: RunState) -> None:
    # A few convenience variables
    dim = network.dim
    hidden_dim = network.hidden_dim
    head_size = dim // network.num_attention_heads

    # Copy the token embedding into x
    state.x = network.weighting.token_embedding_table[token]

    # Pluck out the "pos" row of freq_cis_real and freq_cis_imag
    freq_cis_real_row: NDArray[numpy.float64] = network.weighting.freq_cis_real[step_count]
    freq_cis_imag_row: NDArray[numpy.float64] = network.weighting.freq_cis_imag[step_count]

    # Forward all the layers
    for index_layer in range(network.n_layers):
        # Attention rmsnorm
        state.xb = rmsnorm(state.x, network.weighting.rms_att_weight[index_layer])

        # QKV matmuls for this position
        w = network.weighting.wq[index_layer]
        state.q = numpy.dot(w, state.xb).reshape(network.num_attention_heads, head_size)
        w1 = network.weighting.wk[index_layer]
        state.k = numpy.dot(w1, state.xb).reshape(network.num_attention_heads, head_size)
        w2 = network.weighting.wv[index_layer]
        state.v = numpy.dot(w2, state.xb).reshape(network.num_attention_heads, head_size)

        # Apply RoPE rotation to the q and k vectors for each head
        for index_head in range(network.num_attention_heads):
            # Get the q and k vectors for this head
            q = state.q[index_head]
            k = state.k[index_head]

            # Rotate q and k by the freq_cis_real and freq_cis_imag
            for head_item_index in range(0, head_size, 2):
                q0, q1 = q[head_item_index], q[head_item_index + 1]
                k0, k1 = k[head_item_index], k[head_item_index + 1]
                fcr: numpy.float64 = freq_cis_real_row[head_item_index // 2]
                fci: numpy.float64 = freq_cis_imag_row[head_item_index // 2]
                q[head_item_index] = q0 * fcr - q1 * fci
                q[head_item_index + 1] = q0 * fci + q1 * fcr
                k[head_item_index] = k0 * fcr - k1 * fci
                k[head_item_index + 1] = k0 * fci + k1 * fcr

            # reassigned back to state.q and state.k
            state.q[index_head] = q
            state.k[index_head] = k

        # Save key,value at this time step (pos) to our kv cache
        state.key_cache[step_count, index_layer] = state.k
        state.value_cache[step_count, index_layer] = state.v

        # Multihead attention. Iterate over all heads
        for index_head in range(network.num_attention_heads):
            # Get the query vector for this head
            q = state.q[index_head]

            # Iterate over all timesteps, including the current one
            for timestep in range(step_count + 1):
                # Get the key vector for this head and at this timestep
                k = state.key_cache[timestep, index_layer, index_head]

                # Calculate the attention score as the dot product of q and k
                score = sum(q[i] * k[i] for i in range(head_size))
                score /= math.sqrt(head_size)

                # Save the score to the attention buffer
                state.att[index_head, timestep] = score

            # Softmax the scores to get attention weights, from 0..pos inclusively
            state.att[index_head] = softmax(state.att[index_head], step_count + 1)

            # Weighted sum of the values, store back into xb
            state.xb[index_head * head_size: (index_head + 1) * head_size] = [0.0] * head_size
            for timestep in range(step_count + 1):
                value_vector = state.value_cache[timestep, index_layer, index_head]
                attention_weight: numpy.float64 = state.att[index_head, timestep]
                # Accumulate the weighted value into xb
                for head_item_index in range(head_size):
                    state.xb[index_head * head_size + head_item_index] += attention_weight * value_vector[head_item_index]

        # Final matrix multiplication to get the output of the attention
        state.xb2 = numpy.dot(network.weighting.wo[index_layer], state.xb)

        # Residual connection back into x
        state.x = numpy.add(state.x, state.xb2)

        # Feed-forward Neural Network
        state.xb = rmsnorm(state.x, network.weighting.rms_ffn_weight[index_layer])

        state.hb = numpy.dot(network.weighting.w1[index_layer], state.xb)
        state.hb2 = numpy.dot(network.weighting.w3[index_layer], state.xb)

        sigmoid_linear_unit = numpy.vectorize(lambda value: value / (1. + math.exp(-value)))
        state.hb = numpy.multiply(sigmoid_linear_unit(state.hb), state.hb2)
        state.xb = numpy.dot(network.weighting.w2[index_layer], state.hb)

        # Residual connection
        state.x = numpy.add(state.x, state.xb)

    # Final rmsnorm
    state.x = rmsnorm(state.x, network.weighting.rms_final_weight)

    # Classifier into logits
    state.logits = numpy.dot(network.weighting.wcls, state.x)


def str_lookup(occurrence: str, vocab: List[str]) -> int:
    # Find the first perfect match for string in vocab, return its index or -1 if not found
    try:
        return vocab.index(occurrence)
    except ValueError:
        return -1


def bpe_encode(text: str, vocab: List[str], vocab_scores: NDArray[numpy.float64]) -> List[int]:
    tokens = []

    # First encode every individual character in the input text
    for pos, char in enumerate(text):
        string = char
        pos = str_lookup(string, vocab)
        if pos == -1:
            print(f"not a good prompt at pos {pos}")
            sys.exit(1)
        tokens.append(pos)

    # Merge the best consecutive pair each iteration, according to the scores in vocab_scores
    while True:
        best_score = -1e10
        best_id = -1
        best_idx = -1

        for i in range(len(tokens) - 1):
            # Check if we can merge the pair (tokens[i], tokens[i+1])
            string = vocab[tokens[i]] + vocab[tokens[i + 1]]
            pos = str_lookup(string, vocab)
            if pos != -1 and vocab_scores[pos] > best_score:
                # This merge pair exists in vocab! Record its score and position
                best_score = vocab_scores[pos]
                best_id = pos
                best_idx = i

        if best_idx == -1:
            break  # We couldn't find any more pairs to merge, so we're done

        # Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id
        # Delete token at position best_idx+1, shift the entire sequence back 1
        tokens = tokens[:best_idx + 1] + tokens[best_idx + 2:]

    return tokens


def time_in_ms() -> float:
    # Returns time in milliseconds for benchmarking the model speed
    return int(time.time() * 1000)


def sample(probabilities: NDArray) -> int:
    r = numpy.random.random()
    cdf = numpy.cumsum(probabilities)
    return numpy.argmax(r < cdf)


def run(model_file: BinaryIO, tokenizer_file: BinaryIO, temperature: float, steps: int, prompt: str, seed: int,
        output=sys.stdout):
    if seed is None:
        seed = int(time.time())
    numpy.random.seed(seed)

    # Read in the model.bin file
    network = _load_network(model_file)

    # Right now we cannot run for more than config.seq_len steps
    if steps <= 0 or steps > network.seq_len:
        steps = network.seq_len

    vocab, vocab_scores, max_token_length = tokenizer_init(tokenizer_file, network.vocab_size)

    # Create and initialize the application RunState
    state = _make_init_state(network)

    prompt_tokens = bpe_encode(prompt, vocab, numpy.array(vocab_scores)) if prompt else []
    # Start the main loop
    start: float = 0.  # Used to time our code, only initialized after the first iteration
    # Initialize with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    token_code: int = 1
    pos: int = 0  # Position in the sequence
    # Explicitly print the initial BOS token for stylistic symmetry reasons
    print("<s>", flush=True, file=output)

    while pos < steps:

        # Forward the transformer to get logits for the next token
        transformer(token_code, pos, network, state)

        if pos < len(prompt_tokens):
            # If we are still processing the input prompt, force the next prompt token
            next_token = prompt_tokens[pos]
        elif temperature == 0.0:
            # Greedy argmax sampling: take the token with the highest probability
            next_token = numpy.argmax(state.logits)
        else:
            # Apply the temperature to the logits
            state.logits = numpy.divide(state.logits, temperature)
            # Apply softmax to the logits to get the probabilities for the next token
            state.logits = softmax(state.logits, network.vocab_size)
            # Sample from this distribution to get the next token
            next_token = sample(state.logits)

        # Following BOS token (1), sentencepiece decoder strips any leading whitespace
        token_str = (
            vocab[next_token].lstrip()
            if token_code == 1 and vocab[next_token][0] == ' ' else vocab[next_token]
        )

        print(token_str, end="", flush=True, file=output)

        if next_token == 1:
            break

        # Advance forward
        token_code = next_token
        pos += 1

        # Initialize our timer here because the first iteration could be time-consuming due to IO operations
        if start == 0.:
            start = time_in_ms()

    # Report achieved tok/s
    end = time_in_ms()
    logging.info(f"achieved tok/s: {(steps - 1) / (end - start) * 1000:.01f}")


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
    state.x = numpy.zeros(shape=network.dim)
    state.xb = numpy.zeros(shape=network.dim)
    state.xb2 = numpy.zeros(shape=network.dim)
    state.hb = numpy.zeros(shape=network.hidden_dim)
    state.hb2 = numpy.zeros(shape=network.hidden_dim)
    state.q = numpy.zeros(shape=(network.num_attention_heads, ))
    state.k = numpy.zeros(shape=(network.num_attention_heads, network.head_dimension))
    state.v = numpy.zeros(shape=(network.num_attention_heads, network.head_dimension))
    state.att = numpy.zeros(shape=(network.num_attention_heads, network.seq_len))
    state.logits = numpy.zeros(shape=network.vocab_size)
    state.key_cache = numpy.zeros(shape=(network.seq_len, network.n_layers, network.num_attention_heads, network.head_dimension))
    state.value_cache = numpy.zeros(shape=(network.seq_len, network.n_layers, network.num_attention_heads, network.head_dimension))
    return state
