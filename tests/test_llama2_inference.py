import os.path
import unittest
from pathlib import Path
import io
from typing import List

import numpy
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from llama2 import inference
from llama2.inference import _process_tokens, Network, TransformerWeighting, compute_qkv

_random_magic_number = 0xDEADBEEF


def custom_seed(seed: int) -> None:
    global _random_magic_number
    _random_magic_number = seed


def custom_random() -> float:
    global _random_magic_number
    _random_magic_number = (_random_magic_number * 63) % 0xC4CB7296
    _random_magic_number = _random_magic_number ^ 0x1754FBF
    _random_magic_number = (_random_magic_number * 0xFF) % 4294967296
    _random_magic_number = _random_magic_number ^ 0x222F42CB
    _random_magic_number = _random_magic_number | 0x1234567890
    _random_magic_number = ((_random_magic_number + 14351514) * 32) % 7777333
    return (_random_magic_number % 1000) / 1000.


def generate_random_array(m: int) -> NDArray[numpy.float32]:
    # Create an m by n NumPy array filled with random numbers
    return numpy.array([custom_random() for _ in range(m)], dtype=numpy.float32)


def generate_random_arrays(count: int, m: int) -> List[NDArray[numpy.float32]]:
    # Create an m by n NumPy array filled with random numbers
    return [generate_random_array(m) for _ in range(count + 1)]


def generate_random_matrix(m: int, n: int) -> NDArray[NDArray[numpy.float32]]:
    # Create an m by n NumPy array filled with random numbers
    return numpy.array([[custom_random() for _ in range(n)] for _ in range(m)], dtype=numpy.float32)


def generate_random_matrices(count: int, m: int, n: int) -> List[NDArray[NDArray[numpy.float32]]]:
    # Create an m by n NumPy array filled with random numbers
    return [generate_random_matrix(m, n) for _ in range(count + 1)]


def build_random_network(n_steps: int, n_layers: int, n_vocab: int, head_dimension: int,
                         hidden_dimension: int) -> Network:
    dimension = head_dimension * n_layers
    weighting = TransformerWeighting(
        token_embedding_table=generate_random_matrix(n_vocab, dimension),
        rms_att_weight=generate_random_arrays(n_layers, dimension),
        wq=generate_random_matrices(n_layers, dimension, dimension),
        wk=generate_random_matrices(n_layers, dimension, dimension),
        wv=generate_random_matrices(n_layers, dimension, dimension),
        wo=generate_random_matrices(n_layers, dimension, dimension),
        rms_ffn_weight=generate_random_arrays(n_layers, dimension),
        w1=generate_random_matrices(n_layers, hidden_dimension, dimension),
        w2=generate_random_matrices(n_layers, dimension, hidden_dimension),
        w3=generate_random_matrices(n_layers, hidden_dimension, dimension),
        rms_final_weight=generate_random_array(dimension),
        freq_cis_real=generate_random_arrays(n_steps, head_dimension // 2),
        freq_cis_imag=generate_random_arrays(n_steps, head_dimension // 2)
    )
    return Network(dim=dimension, hidden_dim=hidden_dimension, n_layers=n_layers, num_attention_heads=n_layers,
                   num_key_value_heads=n_layers, vocab_size=n_vocab, seq_len=n_steps,
                   weighting=weighting)


class LLaMa2TestCase(unittest.TestCase):

    def test_find_tokens(self):
        vocab = ['ef', 'cde', 'cd', 'ab', 'abc', 'a', 'b', 'c', 'f', 'fc', 'bcd']
        tokens = list(range(1, len(vocab)))
        vocab_scores = numpy.array([0.5, -0.2, 0.3, 0.1, 0.5, -0.1, 0.4, 0.2, -0.3, 0.7, 0.1])
        result = _process_tokens(tokens, vocab, vocab_scores)
        expected_result = [1, 2, 3, 4, 4, 8, 9, 10]
        self.assertEqual(expected_result, result)

    def test_seed_1(self):
        data_path = str(Path(__file__).parent.parent.absolute())
        with open(os.path.sep.join([data_path, 'data', 'stories15M.bin']), 'rb') as model_file, open(
                os.path.sep.join([data_path, 'data', 'tokenizer.bin']), 'rb') as tokenizer_file:
            prompt = """In that small Swiss town"""
            expected = """<s>
In that small Swiss town, there was a little girl named Lily. She loved to sit in her chair and read her favorite novel. One day, Lily asked her mom if she could help her put some sour candies in her book. 
Her mom said, "Sure, Lily! Let's start by getting some candies." 
Lily went to the kitchen and grabbed a bag of sour candies. She put them in her book and poured the candies into her book. 
Lily's mom smiled and said, "Wow, Lily! You're such a good helper. Did you have fun with your book?" 
Lily replied, "Yes, Mommy! I folded my book and brought some candy to share with you." 
Her mom smiled and said, "That's very kind of you, Lily. You're such a good helper!" 
Lily felt proud of herself and continued to read her books, feeling happy and content.
<s>
"""
            result = io.StringIO()
            inference.run(model_file, tokenizer_file, temperature=0.8, max_steps=256, prompt=prompt, seed=1,
                          output=result)
            self.assertEqual(expected, result.getvalue())

    def test_compute_qkv_small(self):
        custom_seed(2)
        n_vocab = 320
        head_dimension = 8
        n_layers = 3
        network = build_random_network(n_steps=5, n_layers=n_layers, n_vocab=n_vocab, head_dimension=8,
                                       hidden_dimension=2)
        index_layer = 2
        freq_cis_real_row = network.weighting.freq_cis_real[2]
        freq_cis_imag_row = network.weighting.freq_cis_imag[2]
        token = generate_random_array(head_dimension * n_layers)

        # Call the function
        result = compute_qkv(network, index_layer, freq_cis_real_row, freq_cis_imag_row, token)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

        assert_allclose(freq_cis_imag_row, numpy.array([0.629, 0.403, 0.726, 0.048], dtype=numpy.float32), rtol=1e-5)
        assert_allclose(freq_cis_real_row, numpy.array([0.171, 0.255, 0.385, 0.716], dtype=numpy.float32), rtol=1e-5)
        assert_allclose(result[1][1], numpy.array(
            [-1.999097, 5.055076, -1.501158, 3.762684, -2.219901, 6.21974,
                               5.030888, 4.501329], dtype=numpy.float32), rtol=1e-5)
        assert_allclose(result[2][2], numpy.array(
            [6.131495, 5.551599, 5.987549, 5.895988, 6.444849, 6.679024,
                  4.993975, 4.984156], dtype=numpy.float32), rtol=1e-5)

    def test_compute_qkv_full(self):
        custom_seed(2)
        n_vocab = 32000
        head_dimension = 48
        n_layers = 6
        network = build_random_network(n_steps=256, n_layers=n_layers, n_vocab=n_vocab, head_dimension=head_dimension,
                                       hidden_dimension=768)
        index_layer = 4  # Replace with the desired index_layer
        freq_cis_real_row = network.weighting.freq_cis_real[2]
        freq_cis_imag_row = network.weighting.freq_cis_imag[2]
        token = generate_random_array(head_dimension * n_layers)

        # Call the function
        result = compute_qkv(network, index_layer, freq_cis_real_row, freq_cis_imag_row, token)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

        assert_allclose(freq_cis_imag_row[:5], numpy.array([0.346, 0.646, 0.22 , 0.586, 0.981], dtype=numpy.float32), rtol=1e-5)
        assert_allclose(freq_cis_real_row[:5], numpy.array([0.076, 0.564, 0.644, 0.398, 0.813], dtype=numpy.float32), rtol=1e-5)
        assert_allclose(result[1][5][-4:], numpy.array([31.032567, 82.192196, 36.013003, 40.939727], dtype=numpy.float32), rtol=1e-5)

    def test_custom_random(self):
        custom_seed(2)
        self.assertEqual([custom_random() for _ in range(3)], [0.047, 0.453, 0.653])


if __name__ == '__main__':
    unittest.main()
