import os.path
import unittest
from pathlib import Path
import io
from typing import List

import numpy
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from llama2 import inference
from llama2.inference import _process_tokens, Network, TransformerWeighting, compute_qkv, apply_rotations, rms_norm

_random_value = 0xDEADBEEF


def custom_random() -> float:
    global _random_value
    _random_value = (_random_value * 63) % 0xC4CB7296
    _random_value = _random_value ^ 0x1754FBF
    _random_value = (_random_value * 0xFF) % 4294967296
    _random_value = _random_value ^ 0x222F42CB
    _random_value = _random_value | 0x1234567890
    _random_value = ((_random_value + 14351514) * 32) % 7777333
    return (_random_value % 1000) / 1000.


def custom_seed(seed: int) -> None:
    global _random_value
    _random_value = seed


def generate_random_vector(m: int) -> NDArray[numpy.float32]:
    # Create an m by n NumPy array filled with random numbers
    return numpy.array([custom_random() for _ in range(m)], dtype=numpy.float32)


def generate_random_vectors(count: int, m: int) -> List[NDArray[numpy.float32]]:
    # Create an m by n NumPy array filled with random numbers
    return [generate_random_vector(m) for _ in range(count)]


def generate_random_matrix(m: int, n: int) -> NDArray[NDArray[numpy.float32]]:
    # Create an m by n NumPy array filled with random numbers
    return numpy.array([[custom_random() for _ in range(n)] for _ in range(m)], dtype=numpy.float32)


def generate_random_matrices(count: int, m: int, n: int) -> List[NDArray[NDArray[numpy.float32]]]:
    # Create an m by n NumPy array filled with random numbers
    return [generate_random_matrix(m, n) for _ in range(count)]


def build_random_network(n_steps: int, n_layers: int, n_vocab: int, head_dimension: int,
                         hidden_dimension: int) -> Network:
    dimension = head_dimension * n_layers
    weighting = TransformerWeighting(
        token_embedding_table=generate_random_matrix(n_vocab, dimension),
        rms_att_weight=generate_random_vectors(n_layers, dimension),
        wq=generate_random_matrices(n_layers, dimension, dimension),
        wk=generate_random_matrices(n_layers, dimension, dimension),
        wv=generate_random_matrices(n_layers, dimension, dimension),
        wo=generate_random_matrices(n_layers, dimension, dimension),
        rms_ffn_weight=generate_random_vectors(n_layers, dimension),
        w1=generate_random_matrices(n_layers, hidden_dimension, dimension),
        w2=generate_random_matrices(n_layers, dimension, hidden_dimension),
        w3=generate_random_matrices(n_layers, hidden_dimension, dimension),
        rms_final_weight=generate_random_vector(dimension),
        freq_cis_real=generate_random_vectors(n_steps, head_dimension // 2),
        freq_cis_imag=generate_random_vectors(n_steps, head_dimension // 2)
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

    def test_generate_random_vector(self):
        custom_seed(2)
        v = generate_random_vector(4)
        self.assertEqual(v.size, 4)
        self.assertAlmostEqual(float(v[0]), 0.047, places=6)
        self.assertAlmostEqual(float(v[1]), 0.453, places=6)
        self.assertAlmostEqual(float(v[2]), 0.653, places=6)
        self.assertAlmostEqual(float(v[3]), 0.577, places=6)

    def test_generate_random_vectors(self):
        custom_seed(2)
        v1 = generate_random_vector(4)
        self.assertEqual(v1.size, 4)
        self.assertAlmostEqual(float(v1[0]), 0.047, places=6)
        self.assertAlmostEqual(float(v1[1]), 0.453, places=6)
        self.assertAlmostEqual(float(v1[2]), 0.653, places=6)
        self.assertAlmostEqual(float(v1[3]), 0.577, places=6)

        v2 = generate_random_vector(4)
        self.assertAlmostEqual(float(v2[0]), 0.022, places=6)
        self.assertAlmostEqual(float(v2[1]), 0.253, places=6)
        self.assertAlmostEqual(float(v2[2]), 0.432, places=6)
        self.assertAlmostEqual(float(v2[3]), 0.524, places=6)

    def test_build_random_network(self):
        custom_seed(2)
        n_vocab = 320
        n_layers = 3
        network = build_random_network(n_steps=5, n_layers=n_layers, n_vocab=n_vocab, head_dimension=8,
                                       hidden_dimension=2)
        index_layer = 2
        freq_cis_real_row = network.weighting.freq_cis_real[index_layer]
        freq_cis_imag_row = network.weighting.freq_cis_imag[index_layer]
        token_matrix = network.weighting.token_embedding_table
        self.assertAlmostEqual(float(token_matrix[0][0]), 0.047, places=6)
        self.assertAlmostEqual(float(token_matrix[319][23]), 0.828, places=6)

        assert_allclose(freq_cis_real_row, numpy.array([0.828, 0.145, 0.344, 0.043], dtype=numpy.float32), rtol=1e-5)
        assert_allclose(freq_cis_imag_row, numpy.array([0.981, 0.754, 0.745, 0.609], dtype=numpy.float32), rtol=1e-5)
        expected_attn_weights = [0.448, 0.975, 0.957, 0.775, 0.288, 0.913, 0.529, 0.169, 0.7,
                  0.511, 0.013, 0.952, 0.401, 0.661, 0.845, 0.121, 0.272, 0.256,
                  0.376, 0.958, 0.046, 0.471, 0.226, 0.462]
        assert_allclose(network.weighting.rms_att_weight[2], numpy.array(expected_attn_weights, dtype=numpy.float32), rtol=1e-5)

    def test_compute_qkv_small(self):
        custom_seed(2)
        n_vocab = 320
        head_dimension = 8
        n_layers = 3
        network = build_random_network(n_steps=5, n_layers=n_layers, n_vocab=n_vocab, head_dimension=8,
                                       hidden_dimension=2)
        index_layer = 2
        freq_cis_real_row = network.weighting.freq_cis_real[index_layer]
        freq_cis_imag_row = network.weighting.freq_cis_imag[index_layer]
        token = generate_random_vector(head_dimension * n_layers)

        # Call the function
        result = compute_qkv(network, index_layer, freq_cis_real_row, freq_cis_imag_row, token)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

        assert_allclose(freq_cis_imag_row, numpy.array([0.981, 0.754, 0.745, 0.609], dtype=numpy.float32), rtol=1e-5)
        assert_allclose(freq_cis_real_row, numpy.array([0.828, 0.145, 0.344, 0.043], dtype=numpy.float32), rtol=1e-5)
        assert_allclose(result[1][1], numpy.array(
            [-1.262483, 9.873482, -1.809541, 4.85637, -1.716298, 4.831686,
            -2.449315, 3.406103], dtype=numpy.float32), rtol=1e-5)
        assert_allclose(result[2][2], numpy.array(
            [4.61404 , 5.498788, 5.519291, 5.196641, 4.792354, 3.996622,
                  4.755136, 5.863463], dtype=numpy.float32), rtol=1e-5)

    def test_compute_rotations(self):
        custom_seed(2)
        n_vocab = 320
        n_layers = 3
        network = build_random_network(n_steps=5, n_layers=n_layers, n_vocab=n_vocab, head_dimension=8,
                                       hidden_dimension=2)
        index_layer = 2
        wq = numpy.array([5.1699734, 5.4471865, 5.8559923, 6.6097193, 6.1578555, 5.306076, 5.0016994, 6.755227],
                         dtype=numpy.float32)
        freq_cis_real_row = network.weighting.freq_cis_real[index_layer]
        freq_cis_imag_row = network.weighting.freq_cis_imag[index_layer]
        result = apply_rotations(wq, freq_cis_real_row, freq_cis_imag_row)
        assert_allclose(result, numpy.array(
            [-1.062952,  9.582014, -4.13461 ,  5.373827, -1.834724,  6.412893,
                  -3.89886 ,  3.33651],
            dtype=numpy.float32), rtol=1e-5)

    def test_compute_rms_norm(self):
        custom_seed(2)
        n_vocab = 320
        head_dimension = 8
        n_layers = 3
        network = build_random_network(n_steps=5, n_layers=n_layers, n_vocab=n_vocab, head_dimension=8,
                                       hidden_dimension=2)
        token = generate_random_vector(head_dimension * n_layers)
        index_layer = 2
        rba: NDArray[numpy.float32] = rms_norm(token, network.weighting.rms_att_weight[index_layer])
        self.assertAlmostEqual(float(token[0]), 0.445, places=6)
        self.assertAlmostEqual(float(token[23]), 0.529, places=6)
        self.assertAlmostEqual(float(network.weighting.rms_att_weight[index_layer][0]), 0.448, places=6)
        self.assertAlmostEqual(float(network.weighting.rms_att_weight[index_layer][7]), 0.169, places=6)

        self.assertEqual(rba.size, 24)
        self.assertAlmostEqual(float(rba[0]), 0.34457278, places=6)
        self.assertAlmostEqual(float(rba[-1]), 0.42241624, places=6)
        self.assertAlmostEqual(rba.sum(), 9.711192, places=6)

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
        token = generate_random_vector(head_dimension * n_layers)

        # Call the function
        result = compute_qkv(network, index_layer, freq_cis_real_row, freq_cis_imag_row, token)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

        assert_allclose(freq_cis_imag_row[:5], numpy.array([0.769, 0.913, 0.77 , 0.791, 0.171], dtype=numpy.float32),
                        rtol=1e-5)
        assert_allclose(freq_cis_real_row[:5], numpy.array([0.913, 0.529, 0.169, 0.7  , 0.511], dtype=numpy.float32),
                        rtol=1e-5)
        assert_allclose(result[1][5][-4:],
                        numpy.array([ 28.976269, 114.114619, -52.949184,  73.157083], dtype=numpy.float32), rtol=1e-5)

    def test_custom_random(self):
        custom_seed(2)
        self.assertEqual([custom_random() for _ in range(3)], [0.047, 0.453, 0.653])


if __name__ == '__main__':
    unittest.main()
