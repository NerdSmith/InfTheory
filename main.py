from JointEnsemble import JointEnsemble
from Entropy import Entropy
import os

def main():

    # je = JointEnsemble("task3_markov\\input1.txt")


    target_folder = "atta"
    for file in os.listdir(target_folder):
        je = JointEnsemble(target_folder + "\\" + file)
        je.find_probabilities()
        je.find_cond_probabilities()
        je.print_res()
        e = Entropy(je)
        e.calc_binary_entropy()
        e.calc_partial_conditional_entropies()
        e.calc_joint_entropy()
        e.calc_conditional_entropies()
        e.print_res()
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
