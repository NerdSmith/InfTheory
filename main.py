from JointEnsemble import JointEnsemble
from Entropy import Entropy
import os

def main():
    target_folder = "atta_2"
    for file in os.listdir(target_folder):
        print("{:-^50s}".format(file))
        je = JointEnsemble(target_folder + "\\" + file)
        je.draw_graph()
        je.build_eq_system()
        je.balance_eq_system()
        je.print_m_probs()
        je.solve_eq_system()
        je.print_solve_res()
        e = Entropy(je)
        e.calc_simple_entropy()
        e.print_M_be()
        je.calc_join_probabilities()
        je.print_jps()
        e.calc_simple_entropy_with_JP()
        e.print_M_be_with_jp()
        e.calc_cond_entropy()
        e.print_cond_entropy()
        print("{:-^50s}".format("-" * len(file)))

    # target_folder = "atta"
    # for file in os.listdir(target_folder):
    #     je = JointEnsemble(target_folder + "\\" + file)
        # je.find_probabilities()
        # je.find_cond_probabilities()
        # je.print_res()
        # e = Entropy(je)
        # e.calc_binary_entropy()
        # e.calc_partial_conditional_entropies()
        # e.calc_joint_entropy()
        # e.calc_conditional_entropies()
        # e.print_res()
    # print("="*100, "\n"*10, "="*100)
    # target_folder = "task2_entropy_tests"

    # for file in os.listdir(target_folder):
    #     je = JointEnsemble(target_folder + "\\" + file)
    #     je.find_probabilities()
    #     je.find_cond_probabilities()
    #     # je.print_res()
    #     e = Entropy(je)
    #     e.calc_binary_entropy()
    #     e.calc_partial_conditional_entropies()
    #     e.calc_joint_entropy()
    #     e.calc_conditional_entropies()
    #     e.print_res()


if __name__ == '__main__':
    main()
