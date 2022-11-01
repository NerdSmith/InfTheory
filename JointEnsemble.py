from typing import List, Union
import numpy as np
from TaskInpType import TInpType
from Utils import *


class Component:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"{self.name}"

    def __eq__(self, other: "Component"):
        return other.name == self.name


class ComponentIdx:
    def __init__(self, name, idx):
        self.name: str = name
        self.idx: int = idx

    def __str__(self):
        return f"{self.name}_{self.idx}"

    def __eq__(self, other: "ComponentIdx"):
        return other.idx == self.idx and other.name == self.name


class MarkovChProb:
    def __init__(self, prefix, component):
        self.prefix = prefix
        self.component = component

    def component_str(self):
        return self.__str__()

    def __str__(self):
        return f"p({self.prefix} = {self.component})"

    def __str__short__(self):
        return f"p_{self.component}"


class Equation:
    class Term:
        def __init__(self, multiplier: Union["CondProbability", float], m_ch_prob):
            self.multiplier: Union["CondProbability", float] = multiplier
            self.m_ch_prob = m_ch_prob

        def __str__(self):
            return f"{'' if self.multiplier == 1 else str(self.multiplier.val) + ' * '}{self.m_ch_prob}"

    def __init__(self):
        self.res = None
        self.terms = []

    def get_calc_repr_str(self):
        return f"{self.res} = {' + '.join([str(term) for term in self.terms])}"

class EqSystem:
    class Result:
        def __init__(self, m_ch, val):
            self.m_ch = m_ch
            self.val = val

        def __str__(self):
            return f"{self.m_ch.__str__short__()} = {np.round(self.val, 4)}"

    def __init__(self):
        self.eqs = []
        self.matrix_A = []
        self.matrix_B = []
        self.markov_terms: List[MarkovChProb] = []
        self.to_remove_idx = 1 # row nb 2
        self.solve_results = []

    def remove_row(self):
        print(f"Убираем строку {self.to_remove_idx + 1}, потому что ... ДА\n")
        del self.matrix_A[self.to_remove_idx]
        del self.matrix_B[self.to_remove_idx]

    def add_cond(self):
        additional_eq = Equation()
        additional_eq.res = 1
        for term in self.eqs[0].terms:
            m_ch_prob = term.m_ch_prob
            self.markov_terms.append(m_ch_prob)
            additional_eq.terms.append(Equation.Term(1, m_ch_prob))
        self.eqs.append(additional_eq)

    def balance(self):
        self.add_cond()
        for eq in self.eqs:
            if not isinstance(eq.res, int):
                res_component = eq.res.m_ch_prob.component
                # eq.res.multiplier -= 1
                self.matrix_B.append([eq.res.multiplier - 1])
                terms_4_matrix = []
                for term in eq.terms:
                    if term.m_ch_prob.component == res_component:
                        terms_4_matrix.append(term.multiplier.val - 1)
                    else:
                        terms_4_matrix.append(term.multiplier.val)
                self.matrix_A.append(terms_4_matrix)
            else:
                self.matrix_B.append([1])
                terms_4_matrix = []
                for term in eq.terms:
                    terms_4_matrix.append(term.multiplier)
                self.matrix_A.append(terms_4_matrix)

    def solve_system(self):
        self.remove_row()

        AM = copy_matrix(self.matrix_A)
        n = len(self.matrix_A)
        BM = copy_matrix(self.matrix_B)

        print_matrices('Изначальная система:', AM, BM, self.markov_terms)

        indices = list(range(n))
        for fd in range(n):
            fdScaler = 1.0 / AM[fd][fd]

            for j in range(n):
                AM[fd][j] *= fdScaler
            BM[fd][0] *= fdScaler

            string1 = '\nИспользуя систему выше, '
            string1 += 'масштабируем строку-{} (левую и правую части) по '
            string2 = 'диагональному элементу {} левой части, '
            string2 += 'который равен 1/{:+.3f}.\n'
            stringsum = string1 + string2
            val1 = fd + 1
            val2 = fd + 1
            Action = stringsum.format(val1, val2, round(1. / fdScaler, 3))
            print_matrices(Action, AM, BM, self.markov_terms)
            print()

            for i in indices[0:fd] + indices[fd + 1:]:
                crScaler = AM[i][fd]
                for j in range(n):
                    AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
                BM[i][0] = BM[i][0] - crScaler * BM[fd][0]

                string1 = 'Используя систему выше, вычтем {:+.3f} * '
                string1 += '(левую часть ряда-{}) из левой части ряда-{}, и '
                string2 = 'вычтем {:+.3f} * (правую часть ряда-{}) '
                string2 += 'из левой части ряда-{}\n'
                val1 = i + 1
                val2 = fd + 1
                stringsum = string1 + string2
                Action = stringsum.format(crScaler, val2, val1,
                                          crScaler, val2, val1)
                print_matrices(Action, AM, BM, self.markov_terms)

        for i in zip(self.markov_terms, BM):
            self.solve_results.append(self.Result(i[0], i[1][0]))

class JointProbability:
    def __init__(self, components=[], value=0, prob=None, cond_prob=None):
        self.components: List[ComponentIdx] = components
        self.value: float = value

        self.prob = prob
        self.cond_prob = cond_prob
        self.prefixes = ["X_i = ", "X_i+1 = "]

    def val(self):
        return round(self.prob.val * self.cond_prob.val, 3)

    def components_str(self):
        r = "p("
        r += " ".join([str(c) for c in self.components])
        r += ")"
        return r

    def component_str(self):
        r = "p("
        r += ", ".join([c[0] + str(c[1]) for c in zip(self.prefixes, reversed(self.cond_prob.components))])
        r += ")"
        return r

    def get_names_calc_str(self):
        return f"{self.prob.m_ch.__str__short__()} * {self.cond_prob.component_str()}"

    def get_values_calc_str(self):
        return f"{np.round(self.prob.val, 4)} * {self.cond_prob.val}"

    def get_calc_repr_str(self):
        return f"{self.component_str()} = {self.get_names_calc_str()} = {self.get_values_calc_str()} = {self.val()}"

class CondProbability:
    def __init__(self, components, jp=None, p=None, val=None):
        self.components = components
        self.jp = jp
        self.prob = p
        self.val = val

    def value(self):
        return round(self.jp.value / self.prob.value(), 4)

    def get_names_calc_str(self):
        return f"{self.jp.components_str()}/{self.prob.component_str()}"

    def get_values_calc_str(self):
        return f"{self.jp.value}/{self.prob.value()}"

    def component_str(self):
        r = "p("
        r += "|".join([str(c) for c in self.components])
        r += ")"
        return r

    def get_calc_repr_str(self):
        if self.val is not None:
            return f"{self.component_str()} = " \
                   f"{self.val}"
        elif self.jp is not None and self.prob is not None:
            return f"{self.component_str()} = " \
                   f"{self.get_names_calc_str()} = " \
                   f"{self.get_values_calc_str()} = " \
                   f"{np.round(self.value(), 2)}"
        else:
            return f"{self.component_str()} is not set! ERROR"


class Probability:
    def __init__(self, component, terms):
        self.component = component
        self.terms = terms

    def value(self):
        return float(str(round(np.sum(self.terms), 4)))

    def get_calc_str(self):
        terms = [str(round(e, 5)) for e in self.terms.tolist()]
        return " + ".join(terms)

    def component_str(self):
        return f"p({str(self.component)})"

    def get_calc_repr_str(self):
        return f"{self.component_str()} = {self.get_calc_str()} = {str(self.value())}"


class JointEnsemble:

    def __init__(self, path):
        self.filename = None
        self.vars_ = []
        self.vars_max_indices = dict()

        self.table = None
        self.probabilities: List[Probability] = []
        self.cond_probabilities = []
        self.joint_probabilities = []
        self.eq_system = EqSystem()

        self.parse(path)

    def parse(self, path):
        inp = self.read_input(path)
        str_inp_type = inp[0].split("#")[0].strip()

        if str_inp_type == TInpType.CP.value:
            self.parse_file_cp(inp)
        else:
            self.parse_file_jp(inp)

    def read_input(self, filename=None):
        if filename is None:
            pass
        lines = []
        self.filename = filename
        with open(filename, "r") as f:
            lines = f.readlines()

        return lines

    def find_in_CP_by_comp_at_idx(self, comp, idx):
        found = []
        for cp in self.cond_probabilities:
            if cp.components[idx] == comp:
                found.append(cp)
        return found

    def find_in_probabilities(self, prob: Probability):
        for p in self.probabilities:
            if p.component.name == prob.component.name and p.component.idx == prob.component.idx:
                return p
        return None

    def find_cond_probabilities(self):
        self.find_cond_probabilities_inner(self.table, self.vars_, self.joint_probabilities)
        new_vars = self.vars_[:]
        new_vars.reverse()
        self.find_cond_probabilities_inner(self.table.transpose(), new_vars, self.joint_probabilities, True)

    def find_cond_probabilities_inner(self, table, vars_to_add, values_to_add, rev=False):
        curr_var = vars_to_add[0]
        new_vars_to_add = vars_to_add[1:]

        for i in range(self.vars_max_indices[curr_var]):
            table_part = table[i]
            new_values = []
            found_jps = self.find_in_JPs(values_to_add, ComponentIdx(curr_var, i + 1))
            new_values.extend(found_jps)
            if len(new_values) == 1:
                # table[i] = new_values[0].value
                # f_prob = self.find_in_probabilities(Probability(new_values[0].components[-1], []))
                f_prob = self.find_in_probabilities(Probability(ComponentIdx(curr_var, i + 1), []))
                if rev:
                    components = new_values[0].components[:]
                    components.reverse()
                else:
                    components = new_values[0].components
                self.cond_probabilities.append(CondProbability(components, new_values[0], f_prob))
            else:
                self.find_cond_probabilities_inner(table_part.transpose(), new_vars_to_add, new_values, rev)

    def find_probabilities(self):
        self.find_probabilities_inner(self.table, self.vars_)

    def find_probabilities_inner(self, table, vars_to_add):
        curr_var = vars_to_add[0]
        new_vars_to_add = vars_to_add[1:]

        for i in range(self.vars_max_indices[curr_var]):
            table_part = table[i]
            if len(table_part.shape) == 1:
                self.probabilities.append(Probability(ComponentIdx(curr_var, i + 1), table_part))
            else:
                pass
                # TODO: add 3d support
        if len(new_vars_to_add) == 0:
            return
        else:
            self.find_probabilities_inner(table.transpose(), new_vars_to_add) # TODO also add 3d support

    def find_in_res(self, component: Component):
        for res in self.eq_system.solve_results:
            ch = res.m_ch
            if ch.component == component:
                return res
        return None

    def calc_join_probabilities(self):
        for c in self.vars_:
            prob = self.find_in_res(Component(c))
            components_of_cp = self.find_in_CP_by_comp_at_idx(Component(c), 1)
            for inner_c in components_of_cp:
                self.joint_probabilities.append(JointProbability(prob=prob, cond_prob=inner_c))

    def parse_file_cp(self, f_content):
        var_set = set()
        for line in f_content[1:]:
            splitted_line = line.split()
            if splitted_line[1] != "|":
                raise Exception(f"input: {splitted_line} is not correct!")
            first_var = splitted_line[0]
            second_var = splitted_line[2]
            value = float(splitted_line[3])
            var_set.add(first_var)
            var_set.add(second_var)
            cond_prob = CondProbability([Component(first_var), Component(second_var)], val=value)
            self.cond_probabilities.append(cond_prob)
        self.vars_ = sorted(list(var_set))

    def parse_file_jp(self, f_content):
        for line in f_content:
            components = line.split()
            if len(components) < 3:
                raise Exception("err in vars declaration")
            value = float(components[-1])
            content_vars = components[:-1]
            parsed_vars = self.parse_vars(content_vars)
            jp = JointProbability(parsed_vars, value)
            self.joint_probabilities.append(jp)

        self.build_table(self.joint_probabilities)

    def parse_vars(self, content_vars):
        c_vars = []
        for v_i in content_vars:
            v, i = v_i.split("_")
            c_vars.append(ComponentIdx(v, int(i)))
        return c_vars

    def build_eq_system(self):
        for var in self.vars_:
            eq = Equation()
            eq.res = Equation.Term(1.0, MarkovChProb("X_i+1", Component(var)))
            for term_var in self.vars_:
                cp = self.find_in_cond_probabilities([Component(var), Component(term_var)])
                term = Equation.Term(cp, MarkovChProb("X_i", Component(term_var)))
                eq.terms.append(term)
            self.eq_system.eqs.append(eq)

    def build_table(self, joint_probabilities):
        self.fill_vars_and_max_indices(joint_probabilities)

        shape = []
        for entry in self.vars_:
            shape.append(self.vars_max_indices[entry])
        self.table = np.zeros(shape=shape, dtype=np.float32)
        self.set_value(self.table, self.vars_, joint_probabilities)

    def balance_eq_system(self):
        self.eq_system.balance()

    def solve_eq_system(self):
        self.eq_system.solve_system()

    def fill_vars_and_max_indices(self, joint_probabilities: List[JointProbability]):
        for jp in joint_probabilities:
            for component in jp.components:
                # appending vars
                self.vars_.append(component.name)
                self.vars_ = list(dict.fromkeys(self.vars_))

                # appending indices
                if self.vars_max_indices.get(component.name) is None:
                    self.vars_max_indices[component.name] = component.idx
                else:
                    if self.vars_max_indices[component.name] < component.idx:
                        self.vars_max_indices[component.name] = component.idx

    def find_in_JPs(self, jps, component_idx: ComponentIdx):
        new_jps = []
        for jp in jps:
            for component in jp.components:
                if component.name == component_idx.name and component.idx == component_idx.idx:
                    new_jps.append(jp)
        return new_jps

    def set_value(self, table, vars_to_add, values_to_add):
        curr_var = vars_to_add[0]
        new_vars_to_add = vars_to_add[1:]

        for i in range(self.vars_max_indices[curr_var]):
            table_part = table[i]
            new_values = []
            found_jps = self.find_in_JPs(values_to_add, ComponentIdx(curr_var, i + 1))
            new_values.extend(found_jps)
            if len(new_values) == 1:
                table[i] = new_values[0].value
            else:
                self.set_value(table_part, new_vars_to_add, new_values)

    def find_in_cond_probabilities(self, components: List[Union[ComponentIdx, Component]]):
        for cp in self.cond_probabilities:
            if cp.components == components:
                return cp
        return None

    def is_ensembles_dependent(self):
        flag = False # not independent by default

        for jp in self.joint_probabilities:
            r_str = ""
            r_str += jp.components_str()
            r_str += " = "
            mult = 1
            p_vals = []
            for component in jp.components:
                f_prob = self.find_in_probabilities(Probability(component, []))
                p_vals.append(str(f_prob.value()))
                mult *= f_prob.value()
            r_str += "*".join(p_vals)

            mult = np.round(mult, 2)

            if mult != np.round(jp.value, 2):
                r_str += f" = {mult} != "
                flag = True
            else:
                r_str += f" = {mult} = "

            r_str += str(np.round(jp.value, 2))
            print(r_str)

        if flag:
            print("ensembles is dependent")
        else:
            print("ensembles is independent")

    def print_res(self):
        print("{:-^50s}".format(self.filename))
        for p in self.probabilities:
            print(p.get_calc_repr_str())
        print("{:-^50s}".format("-"*len(self.filename)))
        for cp in self.cond_probabilities:
            print(cp.get_calc_repr_str())
        print("{:-^50s}".format("-"*len(self.filename)))
        self.is_ensembles_dependent()
        print("{:-^50s}".format("-"*len(self.filename)))

    def print_m_probs(self):
        for eq in self.eq_system.eqs[:-1]:
            print(eq.get_calc_repr_str())
        print("Добавляем дополнительное условие Σ_s_j∈{a,b,c} p_s_j = 1\n")
        for eq in self.eq_system.eqs:
            print(eq.get_calc_repr_str())

    def print_solve_res(self):
        print("\nЗначения вероятностей стационарного распределения заданной марковской цепи равны:")
        for sr in self.eq_system.solve_results:
            print(sr)

    def print_jps(self):
        print("\nСовместные вероятности:")
        for jp in self.joint_probabilities:
            print(jp.get_calc_repr_str())
