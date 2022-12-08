from typing import List

from Encoding import Encoding
import numpy as np

PRECISION_2 = 15


class Layer:
    def __init__(self, hme: "HMEncoding", prev_layer=None):
        self.hme = hme
        self.prev_layer = prev_layer

        self.idx = 0
        self.curr_prob = None

        self.curr_q = 0
        # invoke fill
        self.fill_by_prev_layer(prev_layer)

        self.curr_G = self.compute_G()
        self.curr_l_m_inner, self.curr_l_m = self.compute_l_m()
        self.curr_G_2 = self.to_code_2(self.curr_G)
        self.code = self.extract_code()

    def fill_by_prev_layer(self, prev_layer: "Layer"):
        if prev_layer is not None:
            self.idx = prev_layer.idx + 1
            self.curr_prob = self.hme.probs[self.idx]
            self.curr_q = np.round(prev_layer.curr_prob.value() + prev_layer.curr_q, 15)
        else:
            self.curr_prob = self.hme.probs[self.idx]

    def compute_G(self):
        return np.round(self.curr_q + self.curr_prob.value() / 2, 15)

    def compute_l_m(self):
        inner_val = -np.log2(self.curr_prob.value())
        return np.round(inner_val, 15), int(np.ceil(inner_val) + 1)

    def get_nb_dec(self, n):
        nb, dec = str(n).split(".")
        return nb, float(f"0.{dec}")

    def to_code_2(self, val):
        val_nb = int(val)
        code_2 = []

        code_10 = val
        for i in range(PRECISION_2):
            mult = code_10 * 2
            nb, dec = self.get_nb_dec(mult)
            code_2.append(nb)
            code_10 = dec

        code_2 = "".join(code_2)
        return f"{np.binary_repr(val_nb)}.{code_2}"

    def extract_code(self):
        _, dec = self.curr_G_2.split(".")
        return dec[:self.curr_l_m]

    def print_as_str(self):
        print(f"{self.curr_prob.name:^5} | "
              f"{self.curr_prob.value():^10} | "
              f"{self.curr_q:^10} | "
              f"{self.curr_G:^10} | "
              f"{np.round(self.curr_l_m_inner, 2):^10} | "
              f"{self.curr_G_2:^20} | "
              f"{self.code:^20} | "
              f"{self.curr_l_m:^3}")


class HMEncoding:
    def __init__(self, probs):
        self.probs = probs
        self.layers: List[Layer] = []

    def build_layers(self):
        prev_layer = None
        for i in range(len(self.probs)):
            new_layer = Layer(self, prev_layer)
            self.layers.append(new_layer)
            prev_layer = new_layer

    def print_layers(self):
        print(f"{'Xm':^5} | {'pm':^10} | {'qm':^10} | {'Gm':^10} | {'lm':^10} | {'G_2':^20} | {'code':^20} | {'Li':^3}")
        for l in self.layers:
            l.print_as_str()

    def get_L(self):
        res = 0
        for l in self.layers:
            res += l.curr_l_m * l.curr_prob.value()
        return np.round(res, 3)

    def get_H(self):
        res = 0
        for l in self.layers:
            res += np.round(l.curr_prob.value() * np.log2(l.curr_prob.value()), 4)
        return np.round(-res, 4)

    def get_K(self):
        res = 0
        for l in self.layers:
            res += np.round(2**(-l.curr_l_m), 4)
        return np.round(res, 4)

    def print_additional(self):
        L = self.get_L()
        H = self.get_H()
        print(f"L={L}")
        print(f"H={H}")
        print(f"r=L-H={L}-{H}={np.round(L-H, 4)} bit")
        print(f"K={self.get_K()}")


if __name__ == '__main__':
    e = Encoding("task5_hilbert_moore\\input1.txt")
    hme = HMEncoding(e.probabilities)
    hme.build_layers()
    hme.print_layers()
    hme.print_additional()
