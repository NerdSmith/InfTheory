import numpy as np
from TaskInpType import TInpType

class Probability:
    def __init__(self, name: str, val: float):
        self.name: str = name
        self.val: float = val

    def value(self):
        return np.round(self.val, 4)


class Encoding:
    def __init__(self, filename):
        self.filename = filename
        self.probabilities = []

        self.parse()

    def read_input(self, filename=None):
        if filename is None:
            pass
        lines = []
        self.filename = filename
        with open(filename, "r") as f:
            lines = f.readlines()

        return lines

    def parse(self):
        inp = self.read_input(self.filename)
        str_inp_type = inp[0].split("#")[0].strip()

        if str_inp_type == TInpType.CODE.value:
            self.parse_file(inp)

    def parse_file(self, inp):
        for line in inp[1:]:
            name, val = line.split()
            val = float(val)
            self.probabilities.append(Probability(name, val))