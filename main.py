from JointEnsemble import JointEnsemble
import os

def main():

    target_folder = "task1_tests"
    for file in os.listdir(target_folder):
        je = JointEnsemble(target_folder + "\\" + file)
        je.find_probabilities()
        je.find_cond_probabilities()
        je.print_res()




if __name__ == '__main__':
    main()
