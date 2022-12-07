from typing import List

import numpy as np

from TaskInpType import TInpType


def extract_digits(s):
    return int(''.join([n for n in s if n.isdigit()]))


class S_i:
    def __init__(self, str_repr):
        self.idx = extract_digits(str_repr)

    def __eq__(self, other):
        return other.idx == self.idx

    def __str__(self):
        return f"s{self.idx}"


class Prob:
    def __init__(self, curr_s_i: S_i, value: float):
        self.curr_s_i = curr_s_i
        self.value = value

    def __str__(self):
        return f"p({str(self.curr_s_i)})={self.value}"

    def str_repr(self):
        return f"p({str(self.curr_s_i)})={self.value}" + int(200 * self.value - len(f"p({str(self.curr_s_i)})={self.value}")) * " "

    def short_str(self, step):
        return f"p({str(self.curr_s_i)})G_{step}" + int(200 * self.value - len(f"p({str(self.curr_s_i)})G_{step}")) * " "

class Q_S_i:
    def __init__(self, curr_idx, curr_Q_S_i: "Q_S_i" = None, curr_prob: Prob = None):
        self.curr_idx = curr_idx
        self.curr_Q_S_i = curr_Q_S_i
        self.curr_prob = curr_prob

    def val(self):
        res = 0
        if self.curr_Q_S_i is not None:
            res += self.curr_Q_S_i.val()
        if self.curr_prob is not None:
            res += self.curr_prob.value
        return res

    def __str__(self):
        if self.curr_Q_S_i is not None:
            return f"q(s{self.curr_idx})={self.curr_Q_S_i.val()}+{self.curr_prob.value}={self.val()}"
        return "0"

    def __eq__(self, other):
        return other.curr_idx == self.curr_idx

    def str_repr(self):
        return int(200 * (self.curr_prob.value if self.curr_prob is not None else 0) - len(f"q(s{self.curr_idx})={self.val()}")) * " " + f"q(s{self.curr_idx})={self.val()}"


class G_S_ik:
    def __init__(self, curr_prob: Prob = None, prev_G_S_ik: "G_S_ik" = None):
        self.curr_prob = curr_prob
        self.prev_G_S_ik = prev_G_S_ik

    def val(self):
        if self.curr_prob is None:
            return 1
        else:
            return np.round(self.curr_prob.value * self.prev_G_S_ik.val(), 16)

    def __str__(self):
        if self.curr_prob is not None:
            return f"{self.curr_prob.value}*{self.prev_G_S_ik.val()}={self.val()}"
        return "1"

    def str_repr(self, step):
        if self.curr_prob is not None:
            return f"G{step}={self.val()}"
        else:
            return None

class F_S_ik:
    def __init__(self,
                 prev_f_s_ik: "F_S_ik" = None,
                 curr_q_s_i: Q_S_i = None,
                 prev_g_s_ik: G_S_ik = None):
        self.prev_f_s_ik = prev_f_s_ik
        self.curr_q_s_i = curr_q_s_i
        self.prev_g_s_ik = prev_g_s_ik

    def val(self):
        if self.prev_f_s_ik is None:
            return 0
        else:
            return self.prev_f_s_ik.val() + self.curr_q_s_i.val() * self.prev_g_s_ik.val()

    def __str__(self):
        if self.prev_f_s_ik is not None:
            return f"{self.prev_f_s_ik.val()}+{self.curr_q_s_i.val()}*{self.prev_g_s_ik.val()}={self.val()}"
        return "0"

    def str_repr(self, step):
        return f"F_{step} = {self.val()}"


class Layer:
    def __init__(self, ae: "AEncoding",  prev_layer=None):
        self.draw_line_len = 200

        self.ae = ae
        self.seq = ae.seq

        self.step = 0
        self.idx = "-"
        self.curr_s_i = None
        self.curr_s_ik = []
        self.curr_p_s_i = None

        self.curr_q_s_i: Q_S_i = self.ae.find_q_s_i_by_idx(1)[0]
        self.curr_F_s_ik = F_S_ik()
        self.curr_G_s_ik = G_S_ik()

        self.fill_by_prev_layer(prev_layer)

    def build(self):
        self.str_step = str(self.step)
        self.str_idx = str(self.idx)
        self.str_s_i = str(self.curr_s_i) if self.curr_s_i is not None else '-'
        self.str_s_ik = ''.join(list(map(str, self.curr_s_ik))) if len(self.curr_s_ik) > 0 else '-'
        self.str_p_s_i = str(self.curr_p_s_i) if self.curr_s_i is not None else '-'
        self.str_q_s_i = str(self.curr_q_s_i)
        self.str_F_s_ik = str(self.curr_F_s_ik)
        self.str_G_s_ik = str(self.curr_G_s_ik)
        self.str_vars = {
            "step": len(self.str_step),
            "idx": len(self.str_idx),
            "s_i": len(self.str_s_i),
            "s_ik": len(self.str_s_ik),
            "p_s_i": len(self.str_p_s_i),
            "q_s_i": len(self.str_q_s_i),
            "F_s_ik": len(self.str_F_s_ik),
            "G_s_ik": len(self.str_G_s_ik)
        }

    def draw(self, compute_x):
        line = ["-" for i in range(self.draw_line_len)]
        line[0] = "|"
        line[-1] = "|"
        prev_val_to_mark = 0
        for p in self.ae.probs:
            val_to_mark = prev_val_to_mark + self.draw_line_len * p.value
            idx_to_mark = int(np.round(val_to_mark)) - 1
            line[idx_to_mark] = "|"
            prev_val_to_mark = val_to_mark

        str_qs = "".join([i.str_repr() for i in self.ae.q_s_i_s])
        if self.step > 0:
            str_probs = "".join([p.short_str(self.step) for p in self.ae.probs])
        else:
            str_probs = "".join([p.str_repr() for p in self.ae.probs])

        x_val = self.get_val_by_code_2()

        min_range_val = self.curr_F_s_ik.val()
        max_range_val = self.curr_F_s_ik.val() + self.curr_G_s_ik.val()
        normalized_x = (compute_x - min_range_val) / (max_range_val - min_range_val)
        val_to_mark = self.draw_line_len * normalized_x
        idx_to_mark = int(np.round(val_to_mark)) - 1
        line[idx_to_mark] = "*"


        # # \r{'k=' + str(self.step):<10}{str(self.curr_s_i if self.curr_s_i else ""):<10}{''.join(line)} i={self.idx:<18} x={x_val}
        print(
            f"""
            \r{'':<20} {f'{str_probs:<{self.draw_line_len}}'}
            # \r{'k=' + str(self.step):<10}{str(self.curr_s_i if self.curr_s_i else ""):<10}{''.join(line)} i={self.idx:<18}
            \r{self.curr_F_s_ik.str_repr(self.step):>20} {f'{self.curr_G_s_ik.str_repr(self.step):^{self.draw_line_len}}' if self.curr_G_s_ik.str_repr(self.step) is not None else f'{str_qs:<{self.draw_line_len}}'} {np.round(self.curr_F_s_ik.val() + self.curr_G_s_ik.val(), 12)}
            """
        )

    def to_str(self, max_len_dict):
        return f"{self.step:{max_len_dict['step']}} | " \
               f"{self.idx:{max_len_dict['idx']}} | " \
               f"{str(self.curr_s_i) if self.curr_s_i is not None else '-':{max_len_dict['s_i']}} | " \
               f"{''.join(list(map(str, self.curr_s_ik))) if len(self.curr_s_ik) > 0 else '-':{max_len_dict['s_ik']}} | " \
               f"{str(self.curr_p_s_i) if self.curr_s_i is not None else '-':{max_len_dict['p_s_i']}} | " \
               f"{str(self.curr_q_s_i):{max_len_dict['q_s_i']}} | " \
               f"{str(self.curr_F_s_ik):{max_len_dict['F_s_ik']}} | " \
               f"{str(self.curr_G_s_ik):{max_len_dict['G_s_ik']}}"

    def get_seq_idx(self):
        idx = self.step - 1
        if idx < 0:
            raise Exception("idx < 0 exception")
        return idx

    def get_s_idx(self):
        return self.curr_s_i.idx

    def fill_by_prev_layer(self, prev_layer: "Layer"):
        if prev_layer is not None:
            self.step = prev_layer.step + 1

            self.curr_s_i = S_i(str(self.seq[self.get_seq_idx()]))

            self.idx = self.curr_s_i.idx

            self.curr_s_ik.extend(prev_layer.curr_s_ik)
            self.curr_s_ik.append(self.curr_s_i)
            self.curr_p_s_i = self.ae.find_p_s_i_by_s_i(self.curr_s_i)

            self.curr_q_s_i = self.ae.find_q_s_i_by_idx(self.get_s_idx())[0]

            self.curr_G_s_ik = G_S_ik(self.curr_p_s_i, prev_layer.curr_G_s_ik)

            self.curr_F_s_ik = F_S_ik(prev_layer.curr_F_s_ik, self.curr_q_s_i, prev_layer.curr_G_s_ik)

    def get_code_val_10(self):
        last_f_s_ik = self.curr_F_s_ik
        last_g_s_ik = self.curr_G_s_ik
        return last_f_s_ik.val() + last_g_s_ik.val() / 2

    def get_nb_dec(self, n):
        nb, dec = str(n).split(".")
        return nb, float(f"0.{dec}")

    def get_code_val_2(self):
        code_10 = self.get_code_val_10()
        str_code_10 = str(code_10)
        max_dec_len = int(self.get_L_val())

        code_2 = []

        for i in range(max_dec_len):
            mult = code_10 * 2
            nb, dec = self.get_nb_dec(mult)
            code_2.append(nb)
            code_10 = dec

        code_2 = "".join(code_2)

        return f"{np.binary_repr(int(code_10))}.{code_2}"

    def get_val_by_code_2(self):
        code_2 = self.get_code_val_2()
        nb, dec = code_2.split(".")
        pow_of_2 = -1
        res = 0
        for d in dec:
            res += 2**pow_of_2 * int(d)
            pow_of_2 -= 1
        return float(f"{res}")

    def get_L_val(self):
        return np.ceil(-np.log2(self.curr_G_s_ik.val())) + 1
        # return 9


class AEncoding:

    def __init__(self, filename):
        self.filename = filename
        self.s_i_s = []
        self.probs = []
        self.q_s_i_s = []
        self.seq = []

        self.layers: List[Layer] = []

        self.parse(filename)
        self.max_i = max([x.idx for x in self.s_i_s])
        self.compute_q_i_s()

    def build_layers(self):
        prev_layer = None
        for i in range(len(self.seq) + 1):
            new_layer = Layer(self, prev_layer)
            new_layer.build()
            self.layers.append(new_layer)
            prev_layer = new_layer
        self.compute_max_val_len()

    def compute_max_val_len(self):
        curr_vals = self.layers[0].str_vars
        for l in self.layers[1:]:
            for key in l.str_vars.keys():
                if curr_vals[key] < l.str_vars[key]:
                    curr_vals[key] = l.str_vars[key]
        self.max_vars_len = curr_vals

    def print_layers(self):
        print(f"{'K':{self.max_vars_len['step']}} | "
              f"{'i':{self.max_vars_len['idx']}} | "
              f"{'Si':{self.max_vars_len['s_i']}} | "
              f"{'Sik':{self.max_vars_len['s_ik']}} | "
              f"{'p(Si)':{self.max_vars_len['p_s_i']}} | "
              f"{'q(Si)':{self.max_vars_len['q_s_i']}} | "
              f"{'F(Sik)':{self.max_vars_len['F_s_ik']}} | "
              f"{'G(Sik)':{self.max_vars_len['G_s_ik']}}"
              )
        for l in self.layers:
            print(l.to_str(self.max_vars_len))

    def draw_layers(self):
        for l in self.layers:
            l.draw(self.layers[-1].get_val_by_code_2())


    def find_q_s_i_by_idx(self, idx):
        return [x for x in self.q_s_i_s if x.curr_idx == idx]

    def compute_q_i_s(self):
        for i in range(self.max_i):
            idx = i + 1
            q_s_i_s = self.find_q_s_i_by_idx(idx - 1)
            p_s_i_s = self.find_p_s_i_by_idx(idx - 1)

            if len(q_s_i_s) == 0:
                q_s_i_s = None
            else:
                q_s_i_s = q_s_i_s[0]
            if len(p_s_i_s) == 0:
                p_s_i_s = None
            else:
                p_s_i_s = p_s_i_s[0]
            new_q_s_i = Q_S_i(idx, q_s_i_s, p_s_i_s)
            self.q_s_i_s.append(new_q_s_i)


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

    def find_p_s_i_by_s_i(self, curr_s_i: S_i):
        return [x for x in self.probs if x.curr_s_i == curr_s_i][0]

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

    def find_p_s_i_by_idx(self, idx):
        return [x for x in self.probs if x.curr_s_i.idx == idx]

    def get_code_val_10(self):
        last_layer = self.layers[-1]
        last_f_s_ik = last_layer.curr_F_s_ik
        last_g_s_ik = last_layer.curr_G_s_ik
        return last_f_s_ik.val() + last_g_s_ik.val() / 2

    def get_nb_dec(self, n):
        nb, dec = str(n).split(".")
        return nb, float(f"0.{dec}")

    def get_code_val_2(self):
        code_10 = self.get_code_val_10()
        str_code_10 = str(code_10)
        max_dec_len = int(self.get_L_val())

        code_2 = []

        for i in range(max_dec_len):
            mult = code_10 * 2
            nb, dec = self.get_nb_dec(mult)
            code_2.append(nb)
            code_10 = dec

        code_2 = "".join(code_2)

        return f"{np.binary_repr(int(code_10))}.{code_2}"

    def get_code(self):
        last_layer = self.layers[-1]
        last_f_s_ik = last_layer.curr_F_s_ik
        last_g_s_ik = last_layer.curr_G_s_ik
        code_10 = self.get_code_val_10()
        code_2 = self.get_code_val_2()
        return f"_x = " \
               f"bin({last_f_s_ik.val()}+{last_g_s_ik.val()}/2) = " \
               f"bin({code_10}) = " \
               f"{code_2}"

    def get_L_val(self):
        last_layer = self.layers[-1]
        last_g_s_ik = last_layer.curr_G_s_ik

        return np.ceil(-np.log2(last_g_s_ik.val())) + 1

    def get_L(self):
        last_layer = self.layers[-1]
        last_g_s_ik = last_layer.curr_G_s_ik

        return f"L = ⌈-log_2(G(_s))⌉ + 1 = ⌈-log_2({last_g_s_ik.val()})⌉ + 1 = {self.get_L_val()}"

class ADecoding:
    def __init__(self, ae):
        self.ae = ae


if __name__ == '__main__':
    ae = AEncoding("task4_a_encoding/input1.txt")
    ae.build_layers()
    ae.print_layers()
    print(ae.get_L())
    print(ae.get_code())
    ae.draw_layers()

    # l1 = Layer(ae)
    # l2 = Layer(ae, l1)
    # l3 = Layer(ae, l2)
    # l4 = Layer(ae, l1)
    # l5 = Layer(ae, l2)
    # l2 = Layer(ae, l1)
    # print(l1.to_str())
    # print(l2.to_str())
    # print(l3.to_str())
    print()
