from typing import List

import numpy as np

from TaskInpType import TInpType


def extract_digits(s):
    return int(''.join([n for n in s if n.isdigit()]))


class S_i:
    def __init__(self, str_repr):
        self.idx = extract_digits(str_repr)


class Prob:
    def __init__(self, curr_s_i: S_i, value: float):
        self.curr_s_i = curr_s_i
        self.value = value

class Q_S_i:
    def __init__(self, curr_Q_S_i: "Q_S_i" = None, curr_prob: Prob = None):
        self.curr_Q_S_i = curr_Q_S_i
        self.curr_prob = curr_prob

    def val(self):
        res = 0
        if self.curr_Q_S_i is not None:
            res += self.curr_Q_S_i.val()
        if self.curr_prob is not None:
            res += self.curr_prob.value
        return res

class AEncoding:

    def __init__(self, filename):
        self.filename = filename
        self.s_i_s = []
        self.probs = []
        self.seq = []

        self.parse(filename)
        self.max_i = max([x.idx for x in self.s_i_s])

    def read_input(self, filename=None):
        if filename is None:
            pass
        lines = []
        self.filename = filename
        with open(filename, "r") as f:
            lines = f.readlines()

        return lines

    def parse(self, path):
        inp = self.read_input(path)
        str_inp_type = inp[0].split("#")[0].strip()

        if str_inp_type == TInpType.A_ENC.value:
            self.parse_file(inp)
        else:
            raise Exception("wrong inp type")

    def find_s_i_by_idx(self, idx):
        return [x for x in self.s_i_s if x.idx == idx][0]

    def parse_file(self, f_content):
        for line in f_content[1:]:
            if line.startswith("s"):
                prep_s_i, prob_val = line.split(" ")
                new_s_i = S_i(prep_s_i)
                new_prob = Prob(new_s_i, float(prob_val))
                self.s_i_s.append(new_s_i)
                self.probs.append(new_prob)
            elif line.startswith("_"):
                for seq_item in line.split()[1:]:
                    extracted = extract_digits(seq_item)
                    found = self.find_s_i_by_idx(extracted)
                    self.seq.append(found)


if __name__ == '__main__':
    ae = AEncoding("task4_a_encoding/input1.txt")
    print()
