import unittest
import torch
from core.quantization import MinMaxQuantizer, LSQQuantizer

class TestQuantization(unittest.TestCase):
    def test_minmax_quantizer(self):
        quant = MinMaxQuantizer(num_states=8)
        x = torch.linspace(-1.5, 1.5, 100)
        y = quant(x)
        self.assertGreaterEqual(y.min().item(), -1.5)
        self.assertLessEqual(y.max().item(), 1.5)
        unique = torch.unique(y)
        self.assertLessEqual(len(unique), 8)

    def test_lsq_quantizer(self):
        quant = LSQQuantizer(num_states=8)
        x = torch.linspace(-1.0, 1.0, 100)
        x.requires_grad = True
        y = quant(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
