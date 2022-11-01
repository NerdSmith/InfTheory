import numpy as np

from JointEnsemble import JointEnsemble, Probability, ComponentIdx


class BinaryEntropy:

    class SumComponent:
        def __init__(self, i, N, probability):
            self.i = i
            self.N = N
            self.probability = probability

        def value(self):
            return self.probability.value() * np.log2(self.probability.value())

        def calc_component_str(self):
            return f"{self.probability.component_str()} * log2({self.probability.component_str()})"

        def calc_value_str(self):
            return f"{self.probability.value()} * log2({self.probability.value()})"

    class MSumComponent:
        def __init__(self, m_ch, val):
            self.m_ch = m_ch
            self.val = val

        def value(self):
            if self.val == 0:
                return 0
            else:
                return np.round(self.val, 4) * np.log2(np.round(self.val, 4))

        def calc_component_str(self):
            return f"{self.m_ch.component_str()} * log2({self.m_ch.component_str()})"

        def calc_value_str(self):
            return f"{np.round(self.val, 4)} * log2({np.round(self.val, 4)})"

    def __init__(self, name):
        self.name = name
        self.sum_components = []

    def value(self):
        return -sum([sum_component.value() for sum_component in self.sum_components])

    def component_str(self):
        return f"H({self.name})"

    def get_components_calc_str(self):
        terms = [s_c.calc_component_str() for s_c in self.sum_components]
        return "-(" + " + ".join(terms) + ")"

    def get_val_calc_str(self):
        terms = [s_c.calc_value_str() for s_c in self.sum_components]
        return "-(" + " + ".join(terms) + ")"

    def get_calc_repr_str(self):
        return f"{self.component_str()} = {self.get_components_calc_str()} = {self.get_val_calc_str()} = {str(round(self.value(), 2))}"


class JointEntropy:
    class SumComponent:
        def __init__(self, joint_probability):
            self.joint_probability = joint_probability

        def value(self):
            return self.joint_probability.value * np.log2(self.joint_probability.value)

        def calc_component_str(self):
            return f"{self.joint_probability.components_str()} * log2({self.joint_probability.components_str()})"

        def calc_value_str(self):
            return f"{self.joint_probability.value} * log2({self.joint_probability.value})"

    def __init__(self, names):
        self.names = names
        self.sum_components = []

    def value(self):
        return -sum([sum_component.value() for sum_component in self.sum_components])

    def component_str(self):
        return f"H({''.join(self.names)})"

    def get_components_calc_str(self):
        terms = [s_c.calc_component_str() for s_c in self.sum_components]
        return "-(" + " + ".join(terms) + ")"

    def get_val_calc_str(self):
        terms = [s_c.calc_value_str() for s_c in self.sum_components]
        return "-(" + " + ".join(terms) + ")"

    def get_calc_repr_str(self):
        return f"{self.component_str()} = {self.get_components_calc_str()} = {self.get_val_calc_str()} = {str(round(self.value(), 2))}"


class PartialConditionalEntropy:
    class SumComponent:
        def __init__(self, conditional_probability):
            self.conditional_probability = conditional_probability

        def value(self):
            return self.conditional_probability.value() * np.log2(self.conditional_probability.value())

        def calc_component_str(self):
            return f"{self.conditional_probability.component_str()} * log2({self.conditional_probability.component_str()})"

        def calc_value_str(self):
            return f"{self.conditional_probability.value()} * log2({self.conditional_probability.value()})"

    def __init__(self, name, cond_name, cond_idx):
        self.name = name
        self.cond_name = cond_name
        self.cond_idx = cond_idx
        self.sum_components = []

    def value(self):
        return -sum([sum_component.value() for sum_component in self.sum_components])

    def pure_value(self):
        return sum([sum_component.value() for sum_component in self.sum_components])

    def component_str(self):
        return f"H{self.cond_name}_{self.cond_idx}({self.name})"

    def get_components_calc_str(self):
        terms = [s_c.calc_component_str() for s_c in self.sum_components]
        return "-(" + " + ".join(terms) + ")"

    def get_val_calc_str(self):
        terms = [s_c.calc_value_str() for s_c in self.sum_components]
        return "-(" + " + ".join(terms) + ")"

    def get_calc_repr_str(self):
        return f"{self.component_str()} = {self.get_components_calc_str()} = {self.get_val_calc_str()} = {str(round(self.value(), 2))}"


class ConditionalEntropy:
    class SumComponent:
        def __init__(self, probability, partial_conditional_entropy):
            self.probability = probability
            self.partial_conditional_entropy = partial_conditional_entropy

        def value(self):
            return self.probability.value() * self.partial_conditional_entropy.value()

        def calc_component_str(self):
            return f"{self.probability.component_str()} * {self.partial_conditional_entropy.component_str()}"

        def calc_value_str(self):
            return f"{self.probability.value()} * {self.partial_conditional_entropy.value()}"

    def __init__(self, name, cond_name):
        self.name = name
        self.cond_name = cond_name
        self.sum_components = []

    def value(self):
        return sum([sum_component.value() for sum_component in self.sum_components])

    def component_str(self):
        return f"H{self.cond_name}({self.name})"

    def get_components_calc_str(self):
        terms = [s_c.calc_component_str() for s_c in self.sum_components]
        return " + ".join(terms)

    def get_val_calc_str(self):
        terms = [s_c.calc_value_str() for s_c in self.sum_components]
        return " + ".join(terms)

    def get_calc_repr_str(self):
        return f"{self.component_str()} = {self.get_components_calc_str()} = {self.get_val_calc_str()} = {str(round(self.value(), 2))}"

class SimpleCondEntropy:
    def __init__(self, binary_e, binary_e_jp):
        self.binary_e: BinaryEntropy = binary_e
        self.binary_e_jp: BinaryEntropy = binary_e_jp

    def value(self):
        return self.binary_e_jp.value() - self.binary_e.value()

    def component_str(self):
        return "H_x_i(X_i+1)"

    def calc_component_str(self):
        return f"{self.binary_e_jp.component_str()} - {self.binary_e.component_str()}"

    def calc_value_str(self):
        return f"{self.binary_e_jp.value()} - {self.binary_e.value()}"

    def get_calc_repr_str(self):
        return f"{self.component_str()} = {self.calc_component_str()} = {self.calc_value_str()} = {self.value()}"

class Entropy:
    def __init__(self, je: JointEnsemble):
        self.je = je
        self.binary_entropies = []
        self.binary_entropies_with_jp = []
        self.joint_entropy = None
        self.partial_conditional_entropies = []
        self.conditional_entropies = []
        self.simple_cond_entropy = None

    def find_in_p_c_e(self, p_c_e: PartialConditionalEntropy):
        for curr_p_c_e in self.partial_conditional_entropies:
            if curr_p_c_e.name == p_c_e.name and curr_p_c_e.cond_name == p_c_e.cond_name and curr_p_c_e.cond_idx == p_c_e.cond_idx:
                return curr_p_c_e
        return None

    def calc_cond_entropy(self):
        self.simple_cond_entropy = SimpleCondEntropy(self.binary_entropies[0], self.binary_entropies_with_jp[0])
        # print(self.simple_cond_entropy.value())

    def calc_simple_entropy(self):
        new_b_entropy = BinaryEntropy("X_i")
        for res in self.je.eq_system.solve_results:
            new_b_entropy.sum_components.append(BinaryEntropy.MSumComponent(res.m_ch, res.val))
        self.binary_entropies.append(new_b_entropy)

    def calc_simple_entropy_with_JP(self):
        new_b_entropy = BinaryEntropy("X_i X_i+1")
        for jp in self.je.joint_probabilities:
            new_b_entropy.sum_components.append(BinaryEntropy.MSumComponent(jp, jp.val()))
        self.binary_entropies_with_jp.append(new_b_entropy)
        # for res in self.je.eq_system.solve_results:
        #     new_b_entropy.sum_components.append(BinaryEntropy.MSumComponent(res.m_ch, res.val))
        # self.binary_entropies.append(new_b_entropy)
        # print(new_b_entropy.get_calc_repr_str())

    def calc_binary_entropy(self):
        for var, max_i in self.je.vars_max_indices.items():
            new_b_entropy = BinaryEntropy(var)
            for i in range(max_i):
                p_to_find = self.je.find_in_probabilities(Probability(ComponentIdx(var, i + 1), []))
                new_b_e_component = BinaryEntropy.SumComponent(i + 1, max_i, p_to_find)
                new_b_entropy.sum_components.append(new_b_e_component)
            self.binary_entropies.append(new_b_entropy)

    def calc_joint_entropy(self):
        self.joint_entropy = JointEntropy(self.je.vars_)
        for jp in self.je.joint_probabilities:
            sum_component = JointEntropy.SumComponent(jp)
            self.joint_entropy.sum_components.append(sum_component)

    def calc_partial_conditional_entropies(self):
        for var in self.je.vars_:
            cond_name = [v for v in self.je.vars_ if v != var][0]
            for curr_cond_idx in range(self.je.vars_max_indices[cond_name]):
                p_c_e = PartialConditionalEntropy(var, cond_name, curr_cond_idx + 1)
                max_var_i = self.je.vars_max_indices[var]
                for curr_var_idx in range(max_var_i):
                    curr_c_e = self.je.find_in_cond_probabilities(
                        [ComponentIdx(var, curr_var_idx + 1), ComponentIdx(cond_name, curr_cond_idx + 1)]
                    )
                    sum_component = PartialConditionalEntropy.SumComponent(curr_c_e)
                    p_c_e.sum_components.append(sum_component)
                self.partial_conditional_entropies.append(p_c_e)

    def calc_conditional_entropies(self):
        for var in self.je.vars_:
            cond_name = [v for v in self.je.vars_ if v != var][0]
            ce = ConditionalEntropy(var, cond_name)
            for curr_cond_idx in range(self.je.vars_max_indices[cond_name]):
                prob = self.je.find_in_probabilities(Probability(ComponentIdx(cond_name, curr_cond_idx + 1), []))
                curr_p_c_e = self.find_in_p_c_e(PartialConditionalEntropy(var, cond_name, curr_cond_idx + 1))
                ce.sum_components.append(ConditionalEntropy.SumComponent(prob, curr_p_c_e))
            self.conditional_entropies.append(ce)

    def print_M_be(self):
        print("\nЭнтропия H(X_i):")
        print(self.binary_entropies[0].get_calc_repr_str())

    def print_M_be_with_jp(self):
        print("\nЭнтропия H(X_i X_i+1):")
        print(self.binary_entropies_with_jp[0].get_calc_repr_str())

    def print_cond_entropy(self):
        print("\nУсловная вероятность H_x_i(X_i+1)")
        print(self.simple_cond_entropy.get_calc_repr_str())


    def print_res(self):
        print("{:-^50s}".format(self.je.filename))
        for be in self.binary_entropies:
            print(be.get_calc_repr_str())
        print("{:-^50s}".format("-" * len(self.je.filename)))
        print(self.joint_entropy.get_calc_repr_str())
        print("{:-^50s}".format("-" * len(self.je.filename)))
        for p_c_e in self.partial_conditional_entropies:
            print(p_c_e.get_calc_repr_str())
        print("{:-^50s}".format("-" * len(self.je.filename)))
        for c_e in self.conditional_entropies:
            print(c_e.get_calc_repr_str())
        print("{:-^50s}".format("-" * len(self.je.filename)))


def main():
    target_folder = "task2_entropy_tests"
    je = JointEnsemble(target_folder + "\\" + "input1.txt")
    je.find_probabilities()
    je.find_cond_probabilities()
    je.print_res()
    e = Entropy(je)
    e.calc_binary_entropy()
    e.calc_partial_conditional_entropies()
    e.calc_joint_entropy()
    e.calc_conditional_entropies()
    e.print_res()
    print()


if __name__ == '__main__':
    main()
