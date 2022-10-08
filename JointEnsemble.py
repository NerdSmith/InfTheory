from typing import List
import numpy as np


class ComponentIdx:
    def __init__(self, name, idx):
        self.name: str = name
        self.idx: int = idx

    def __str__(self):
        return f"{self.name}_{self.idx}"

    def __eq__(self, other: "ComponentIdx"):
        return other.idx == self.idx and other.name == self.name


class JointProbability:
    def __init__(self, components, value):
        self.components: List[ComponentIdx] = components
        self.value: float = value

    def components_str(self):
        r = "p("
        r += " ".join([str(c) for c in self.components])
        r += ")"
        return r


class CondProbability:
    def __init__(self, components, jp, p):
        self.components = components
        self.jp = jp
        self.prob = p

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
        return f"{self.component_str()} = " \
               f"{self.get_names_calc_str()} = " \
               f"{self.get_values_calc_str()} = " \
               f"{np.round(self.value(), 2)}"


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

        self.parse(path)

    def parse(self, path):
        inp = self.read_input(path)
        self.parse_file(inp)

    def read_input(self, filename=None):
        if filename is None:
            pass
        lines = []
        self.filename = filename
        with open(filename, "r") as f:
            lines = f.readlines()

        return lines

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

    def parse_file(self, f_content):

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

    def build_table(self, joint_probabilities):
        self.fill_vars_and_max_indices(joint_probabilities)

        shape = []
        for entry in self.vars_:
            shape.append(self.vars_max_indices[entry])
        self.table = np.zeros(shape=shape, dtype=np.float32)
        self.set_value(self.table, self.vars_, joint_probabilities)

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

    def find_in_cond_probabilities(self, components: List[ComponentIdx]):
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
