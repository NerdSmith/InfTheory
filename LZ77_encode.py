

class LZ77:
    def __init__(self, text):
        self.text = text
        self.search_buf_log = []
        self.inp_buf_log = []
        self.code_words = []

    def l_to_str(self, l):
        return "".join(l)

    def get_red(self, s):
        return f"\033[31m{s}\033[0m"

    def encode(self):
        input = list(self.text)
        find_buf = []
        counter = 1

        print(f"{'N':2} | {'SEARCH BUF':{len(self.text)}} | {'PRE INPUT':{len(self.text)}} | {'CODEWORD':11}")

        while input:

            prev_find_buf = find_buf[:]
            input, idx_start, to_grep_list = self.grep(input, find_buf, 1)
            # print("".join(input), idx_start, to_grep_list, prev_find_buf)
            if idx_start == -1:
                self.code_words.append((0, 0, to_grep_list[-1]))
                vec = (0, 0, to_grep_list[-1])
            else:
                x = len(prev_find_buf) - idx_start
                y = len(to_grep_list) - 1
                if x * y == 0:
                    x = 0
                    y = 0
                vec = (x, y, to_grep_list[-1])
                self.code_words.append(vec)
            pre_inp_str = self.get_red(self.l_to_str(to_grep_list)) + ''.join(input) + " " * (len(self.text) - len(to_grep_list) - len(input))
            print(f"{counter:2} | {self.l_to_str(prev_find_buf):{len(self.text)}} | {pre_inp_str} | {str(vec):11}")
            counter += 1
        print(f"{counter:2} | {self.l_to_str(list(self.text)):{len(self.text)}} | {'':{len(self.text)}} | ")

    def grep(self, input, find_buf, to_grep):

        to_grep_list = input[:to_grep]

        res = False
        found_idx_start = -1

        for idx in range(len(find_buf) - len(to_grep_list) + 1):
            if find_buf[idx: idx + len(to_grep_list)] == to_grep_list:
                res = True
                found_idx_start = idx
                break

        if not res:
            find_buf.extend(to_grep_list)
        else:
            if len(input) < to_grep + 1:
                return input[to_grep:], found_idx_start, to_grep_list
            inp, f_idx_start, to_g_list = self.grep(input, find_buf, to_grep + 1)
            if f_idx_start != -1 and f_idx_start != found_idx_start:
                found_idx_start = f_idx_start
            return inp, found_idx_start, to_g_list


        return input[to_grep:], found_idx_start, to_grep_list

    def print_code_words(self):
        for cw in self.code_words:
            print(cw)


if __name__ == '__main__':
    lz77 = LZ77("У_ОСЫ_НЕ_УСЫ_И_НЕ_УСИЩА,_А_УСИКИ")
    lz77.encode()