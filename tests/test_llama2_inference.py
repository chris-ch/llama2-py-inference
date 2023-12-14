import os.path
import unittest
from pathlib import Path
import io

import numpy

from llama2 import inference
from llama2.inference import _process_tokens


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
            inference.run(model_file, tokenizer_file, temperature=0.8, max_steps=256, prompt=prompt, seed=1, output=result)
            self.assertEqual(expected, result.getvalue())


if __name__ == '__main__':
    unittest.main()
