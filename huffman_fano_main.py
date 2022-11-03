from Encoding import Encoding
from Huffman import HuffGraph
from Fano import FanoGraph
import os

def main():
    target_folder = "task3_huffman_fano"
    for file in os.listdir(target_folder):
        print("{:-^50s}".format(file))
        e = Encoding(target_folder + "\\" + file)
        print("Huffman")
        hg = HuffGraph(e.probabilities)
        print("Fano")
        fg = FanoGraph(e.probabilities)
        print("{:-^50s}".format("-" * len(file)))



if __name__ == '__main__':
    main()