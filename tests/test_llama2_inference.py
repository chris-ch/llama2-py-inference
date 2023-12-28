import os.path
import unittest
from pathlib import Path
import io
from typing import List

import numpy
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from llama2 import inference
from llama2.inference import _process_tokens, Network, TransformerWeighting, compute_qkv, apply_rotations, rms_norm, \
    multihead_activation, build_activation, compute_delta_ffn, create_layer_token

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
        assert_allclose(network.weighting.rms_att_weight[2], numpy.array(expected_attn_weights, dtype=numpy.float32),
                        rtol=1e-5)

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
        assert_allclose(result[0][2], numpy.array(
            [-0.58764449, 9.89856247, -3.21608903, 3.75628453, -2.92128194, 5.04793915, -3.18034321, 3.72614302],
            dtype=numpy.float32), rtol=1e-5)
        assert_allclose(result[1][1], numpy.array(
            [-1.262483, 9.873482, -1.809541, 4.85637, -1.716298, 4.831686,
             -2.449315, 3.406103], dtype=numpy.float32), rtol=1e-5)
        assert_allclose(result[2][2], numpy.array(
            [4.61404, 5.498788, 5.519291, 5.196641, 4.792354, 3.996622,
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
            [-1.062952, 9.582014, -4.13461, 5.373827, -1.834724, 6.412893, -3.89886, 3.33651],
            dtype=numpy.float32), rtol=1e-5)

    def test_multihead_activations(self):
        custom_seed(2)
        n_vocab = 320
        n_layers = 4
        network = build_random_network(n_steps=5, n_layers=n_layers, n_vocab=n_vocab, head_dimension=48,
                                       hidden_dimension=2)
        index_layer = 2
        heads_q = [generate_random_vector(48) for _ in range(6)]
        expected_hq1 = [0.734, 0.616, 0.897, 0.159, 0.346, 0.646, 0.22, 0.586, 0.981, 0.769, 0.913, 0.77, 0.791, 0.171,
                        0.255, 0.385,
                        0.716, 0.948, 0.233, 0.858, 0.206, 0.161, 9.0e-2, 0.195, 0.828, 0.145, 0.344, 4.3e-2, 0.766,
                        0.949, 0.75, 0.7,
                        0.953, 0.514, 0.37, 0.866, 0.755, 0.629, 0.403, 0.726, 4.8e-2, 0.821, 0.872, 0.752, 0.981,
                        0.754, 0.745, 0.609]
        expected_hq6 = [9.0e-2, 0.195, 0.828, 0.145, 0.344, 4.3e-2, 0.766, 0.949, 0.75, 0.7, 0.953, 0.514, 0.37, 0.866,
                        0.755, 0.629,
                        0.403, 0.726, 4.8e-2, 0.821, 0.872, 0.752, 0.981, 0.754, 0.745, 0.609, 0.162, 7.6e-2, 0.564,
                        0.644, 0.398, 0.813,
                        0.421, 0.665, 0.445, 0.391, 0.504, 0.73, 0.434, 0.32, 0.323, 0.323, 0.483, 0.502, 0.984, 0.14,
                        9.0e-2, 0.232]
        assert_allclose(heads_q[0], numpy.array(expected_hq1), rtol=1e-5)
        assert_allclose(heads_q[-1], numpy.array(expected_hq6), rtol=1e-5)
        key_cache = [[[generate_random_vector(48) for _ in range(6)] for _ in range(6)],
                     [[generate_random_vector(48) for _ in range(6)] for _ in range(3)]]
        value_cache = [[[generate_random_vector(48) for _ in range(6)] for _ in range(6)],
                       [[generate_random_vector(48) for _ in range(6)] for _ in range(3)]]

        head_scores_example = numpy.array([0.5194815185588364, 0.48051848144116366])
        activation = build_activation(network.head_dimension, 2, value_cache, 3, head_scores_example)

        expected_activation = numpy.array([0.3045971, 0.28449363, 0.48838997, 0.26805186, 0.72583091,
                                           0.58409174, 0.67818201, 0.68331219, 0.7507793, 0.48202663,
                                           0.26566214, 0.45681779, 0.32925986, 0.72464937, 0.78846788,
                                           0.55206428, 0.5221176, 0.27327259, 0.3940515, 0.15246741,
                                           0.38288274, 0.90151936, 0.44484355, 0.61741503, 0.39233694,
                                           0.77801296, 0.57515665, 0.51214337, 0.54863667, 0.83911714,
                                           0.72254506, 0.30416898, 0.86215585, 0.49119536, 0.40411736,
                                           0.25259773, 0.47084469, 0.42280443, 0.49616951, 0.61828625,
                                           0.41131239, 0.87768853, 0.84770113, 0.74740264, 0.65272719,
                                           0.54209012, 0.28646711, 0.47077943], dtype=numpy.float32)
        assert_allclose(expected_activation, activation, rtol=1e-5)

        result = multihead_activation(network, index_layer, key_cache, value_cache, heads_q)
        result_1 = numpy.array([0.30273916, 0.64124675, 0.43411668, 0.313628, 0.60880145,
                                0.72886318, 0.07149604, 0.55496394, 0.32315882, 0.43760963,
                                0.83072144, 0.31905746, 0.35306539, 0.58717048, 0.64360785,
                                0.87371045, 0.15746488, 0.67458463, 0.3655614, 0.32704458,
                                0.44000856, 0.40689553, 0.17859619, 0.91154489, 0.26830728,
                                0.6173085, 0.6238455, 0.44949539, 0.20511425, 0.31641296,
                                0.53728098, 0.58635251, 0.41710811, 0.54921317, 0.58793827,
                                0.29856142, 0.28704336, 0.49492364, 0.26605987, 0.72003424,
                                0.60054549, 0.68194684, 0.6928338, 0.75157607, 0.49483505,
                                0.26173794, 0.44845147, 0.33157054], dtype=numpy.float32)
        result_2 = numpy.array([0.72445679, 0.79201955, 0.5449608, 0.52959546, 0.27177485,
                                0.38886295, 0.15107666, 0.37869981, 0.8998825, 0.43816797,
                                0.61132783, 0.38455947, 0.77774546, 0.58347964, 0.51743884,
                                0.55200651, 0.84165254, 0.71790206, 0.30563458, 0.86224139,
                                0.49706853, 0.40912402, 0.25594619, 0.47652531, 0.41445997,
                                0.50340134, 0.62393476, 0.41889724, 0.87998855, 0.845615,
                                0.74734908, 0.65175366, 0.53362799, 0.28178137, 0.47285482,
                                0.74830011, 0.56572589, 0.72655302, 0.41254321, 0.69365904,
                                0.25848126, 0.59302279, 0.67689392, 0.74033603, 0.69099176,
                                0.52390209, 0.52126426, 0.45736867], dtype=numpy.float32)
        result_3 = numpy.array([0.31022703, 0.27271248, 0.7579419, 0.41126971, 0.25577594,
                                0.54471373, 0.30786723, 0.73380324, 0.49394724, 0.41732068,
                                0.72149202, 0.83482099, 0.59146233, 0.30522346, 0.59544219,
                                0.41812387, 0.33800535, 0.59342974, 0.76558447, 0.08135566,
                                0.48804265, 0.31003851, 0.39995672, 0.82831475, 0.35244043,
                                0.36672913, 0.64298998, 0.68871374, 0.85950324, 0.17625252,
                                0.70889923, 0.38908476, 0.30429757, 0.47742857, 0.42187907,
                                0.15476229, 0.91736749, 0.28740546, 0.62359694, 0.6556758,
                                0.40221577, 0.21474098, 0.30771784, 0.51406814, 0.63984291,
                                0.36369531, 0.53795612, 0.53584526], dtype=numpy.float32)
        result_4 = numpy.array([0.3045971, 0.28449363, 0.48838997, 0.26805186, 0.72583091,
                                0.58409174, 0.67818201, 0.68331219, 0.7507793, 0.48202663,
                                0.26566214, 0.45681779, 0.32925986, 0.72464937, 0.78846788,
                                0.55206428, 0.5221176, 0.27327259, 0.3940515, 0.15246741,
                                0.38288274, 0.90151936, 0.44484355, 0.61741503, 0.39233694,
                                0.77801296, 0.57515665, 0.51214337, 0.54863667, 0.83911714,
                                0.72254506, 0.30416898, 0.86215585, 0.49119536, 0.40411736,
                                0.25259773, 0.47084469, 0.42280443, 0.49616951, 0.61828625,
                                0.41131239, 0.87768853, 0.84770113, 0.74740264, 0.65272719,
                                0.54209012, 0.28646711, 0.47077943], dtype=numpy.float32)
        assert_allclose(result[0], result_1, rtol=1e-5)
        assert_allclose(result[1], result_2, rtol=1e-5)
        assert_allclose(result[2], result_3, rtol=1e-5)
        assert_allclose(result[3], result_4, rtol=1e-5)

    def test_apply_rotations(self):
        head_vector = numpy.array([1.0, 2.0, 3.0, 4.0])
        freq_cis_real_row = numpy.array([0.5, 0.2])
        freq_cis_imag_row = numpy.array([0.8, 0.3])
        result = apply_rotations(head_vector, freq_cis_real_row, freq_cis_imag_row)
        assert_allclose(result, numpy.array(
            [-1.1, 1.8, -0.6, 1.7],
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
        self.assertAlmostEqual(float(rba[0]), 0.3445728, places=6)
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

        assert_allclose(freq_cis_imag_row[:5], numpy.array([0.769, 0.913, 0.77, 0.791, 0.171], dtype=numpy.float32),
                        rtol=1e-5)
        assert_allclose(freq_cis_real_row[:5], numpy.array([0.913, 0.529, 0.169, 0.7, 0.511], dtype=numpy.float32),
                        rtol=1e-5)
        assert_allclose(result[1][5][-4:],
                        numpy.array([28.976269, 114.114619, -52.949184, 73.157083], dtype=numpy.float32), rtol=1e-5)

    def test_custom_random(self):
        custom_seed(2)
        self.assertEqual([custom_random() for _ in range(3)], [0.047, 0.453, 0.653])

    def test_reshape(self):
        custom_seed(2)
        matrix = generate_random_matrix(3, 4)
        assert_allclose(matrix, numpy.array([
            [0.047, 0.453, 0.653, 0.577],
             [0.022, 0.253, 0.432, 0.524],
             [0.114, 0.917, 0.747, 0.164]
        ], dtype=numpy.float32), rtol=1e-5)
        assert_allclose(matrix.reshape(12), numpy.array([0.047, 0.453, 0.653, 0.577,
                                                         0.022, 0.253, 0.432, 0.524,
                                                         0.114, 0.917, 0.747, 0.164], dtype=numpy.float32), rtol=1e-5)

    def test_compute_delta_ffn(self):
        custom_seed(2)
        n_vocab = 32000
        head_dimension = 48
        n_layers = 6
        network = build_random_network(n_steps=256, n_layers=n_layers, n_vocab=n_vocab, head_dimension=head_dimension,
                                       hidden_dimension=768)
        index_layer = 4  # Replace with the desired index_layer
        token = generate_random_vector(288)
        delta_ffn = compute_delta_ffn(network.weighting, index_layer, token)
        self.assertEqual(len(delta_ffn), 288)
        self.assertAlmostEqual(float(token[0]), 0.616, places=6)
        self.assertAlmostEqual(float(token[287]), 0.176, places=6)
        self.assertAlmostEqual(float(delta_ffn[0]), 1749410.1171746738, places=6)
        self.assertAlmostEqual(float(delta_ffn[287]), 1736456.487817695, places=6)
        self.assertAlmostEqual(float(sum(delta_ffn)), 500590281.58948934, places=6)
        self.assertAlmostEqual(float(min(delta_ffn)), 1723942.4036516517, places=6)
        self.assertAlmostEqual(float(max(delta_ffn)), 1753680.2377535875, places=6)

    def test_create_layer_token(self):
        custom_seed(2)
        n_vocab = 32000
        head_dimension = 48
        n_layers = 6
        network = build_random_network(n_steps=256, n_layers=n_layers, n_vocab=n_vocab, head_dimension=head_dimension,
                                       hidden_dimension=768)
        index_layer = 2
        step_count = 2
        token = generate_random_vector(288)
        freq_cis_real_row = network.weighting.freq_cis_real[2]
        freq_cis_imag_row = network.weighting.freq_cis_imag[2]

        key_cache = [
            [[generate_random_vector(48) for _ in range(6)] for _ in range(6)],
            [[generate_random_vector(48) for _ in range(6)] for _ in range(6)],
            [[generate_random_vector(48) for _ in range(6)] for _ in range(2)]
        ]
        value_cache = [
            [[generate_random_vector(48) for _ in range(6)] for _ in range(6)],
            [[generate_random_vector(48) for _ in range(6)] for _ in range(6)],
            [[generate_random_vector(48) for _ in range(6)] for _ in range(2)]
        ]
        new_token, new_kc, new_vc = create_layer_token(network, step_count, key_cache, value_cache, index_layer,
                                             freq_cis_real_row, freq_cis_imag_row, token)

        self.assertEqual(len(key_cache[2]), 3)
        self.assertEqual(len(value_cache[2]), 3)
        self.assertAlmostEqual(float(key_cache[2][2][0][0]), 13.5140090, places=6)
        self.assertAlmostEqual(float(key_cache[2][2][5][47]), 74.42684423, places=6)
        self.assertAlmostEqual(float(value_cache[2][2][0][0]), 69.27178955, places=6)
        self.assertAlmostEqual(float(value_cache[2][2][5][47]), 68.813262939, places=6)
        self.assertEqual(len(new_token), 288)
        self.assertAlmostEqual(float(token[0]), 0.616, places=6)
        self.assertAlmostEqual(float(token[287]), 0.176, places=6)
        self.assertAlmostEqual(float(new_token[0]), 2439003.2108297, places=6)
        self.assertAlmostEqual(float(new_token[287]), 2442824.50969825, places=6)
        self.assertAlmostEqual(float(sum(new_token)), 701135654.4605024, places=6)
        self.assertAlmostEqual(float(min(new_token)), 2418155.80873464, places=6)
        self.assertAlmostEqual(float(max(new_token)), 2453978.8297747127, places=6)


if __name__ == '__main__':
    unittest.main()
