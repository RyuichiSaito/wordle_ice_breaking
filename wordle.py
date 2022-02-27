import pandas as pd
import numpy as np
import warnings
import random

from collections import Counter


# ignore warnings
warnings.simplefilter('ignore')


class WordleData:
    def __init__(self, filename):
        self.filename = filename
        self.df_five_unique = None
        self.df_five = None
        self.df_five_sep = None

    def read_data(self):
        """
        Reads words from the file and creates wordle format dataframe.
        """
        df = pd.read_table(self.filename, sep='\t', header=None, names=['word', 'freq'])
        df['word'] = df['word'].str.replace('[^a-z]', '')  # remove all non-alphabetical characters
        self.df_five = df[df['word'].str.len() == 5]  # select only words of length 5
        self.df_five['word'] = self.df_five['word'].str.upper()  # lowercase all words and get unique words
        self.df_five = self.df_five.drop_duplicates(subset='word', keep='first')  # remove duplicates
        # remove words with more than one unique character
        self.df_five_unique = self.df_five[self.df_five['word'].apply(lambda x: len(set(x))) == 5]
        self.df_five_unique = self.df_five_unique.sort_values(by='freq').reset_index(drop=True)

    def read_data_nltk(self):
        """
        Reads words from the file and creates wordle format dataframe.
        """
        self.df_five_unique = pd.read_table("wordle_dict.txt", sep='\t', header=None, names=['word'])
        self.df_five = pd.read_table("wordle_dict_all_2.txt", sep='\t', header=None, names=['word'])
        self.df_five_unique['word'] = self.df_five_unique['word'].str.upper()  # lowercase all words and get unique words
        self.df_five['word'] = self.df_five['word'].str.upper()  # lowercase all words and get unique words

    def choice_word(self):
        idx = random.randint(0, self.df_five.shape[0] - 1)
        answord = self.df_five.iloc[idx, 0]
        print("The anser word is: ", answord)
        return answord

    def serch_word(self, word):
        """
        Search if a word is included or not. return >= bool
        """
        return self.df_five['word'].str.contains(word).any()

    def df_sep(self):
        """
        Separate words into columns.
        """
        self.df_five_sep = self.df_five.copy()
        self.df_five_sep['word_1'] = self.df_five_sep['word'].str[0]
        self.df_five_sep['word_2'] = self.df_five_sep['word'].str[1]
        self.df_five_sep['word_3'] = self.df_five_sep['word'].str[2]
        self.df_five_sep['word_4'] = self.df_five_sep['word'].str[3]
        self.df_five_sep['word_5'] = self.df_five_sep['word'].str[4]
        self.df_five_sep_delnot = self.df_five_sep.copy()
        # print(self.df_five_sep.head)

    def print_five_sep(self):
        print(self.df_five_sep)
        self.calc_entropy_all()

    def del_include(self):
        """
        get include words
        """
        while self.usr_included:
            w = self.usr_included.pop()
            self.df_five_sep = self.df_five_sep.query('word_1 == @w or word_2 == @w or word_3 == @w or word_4 == @w or word_5 == @w')
            self.df_five_sep = self.df_five_sep.reset_index(drop=True)

    def del_notinclude(self):
        """
        delete not indlude words
        """
        while self.usr_notindcluded:
            w = self.usr_notindcluded.pop()
            self.df_five_sep = self.df_five_sep.query('word_1 != @w and word_2 != @w and word_3 != @w and word_4 != @w and word_5 != @w')
            self.df_five_sep = self.df_five_sep.reset_index(drop=True)
            self.df_five_sep_delnot = self.df_five_sep_delnot.query('word_1 != @w and word_2 != @w and word_3 != @w and word_4 != @w and word_5 != @w')
            self.df_five_sep_delnot = self.df_five_sep_delnot.reset_index(drop=True)

    def del_decision(self):
        """
        get include words
        """
        col = ['word_1', 'word_2', 'word_3', 'word_4', 'word_5']
        for i, w in enumerate(self.usr_decision):
            if not w == "":
                self.df_five_sep = self.df_five_sep[self.df_five_sep[col[i]] == w]
                self.df_five_sep = self.df_five_sep.reset_index(drop=True)

    def del_notdecision(self):
        """
        del notdecision
        """
        for i in range(5):
            while self.usr_notdecision[i]:
                w = self.usr_notdecision[i].pop()
                self.df_five_sep = self.df_five_sep[self.df_five_sep['word_' + str(i + 1)] != w]
                self.df_five_sep = self.df_five_sep.reset_index(drop=True)
                self.df_five_sep_delnot = self.df_five_sep_delnot[self.df_five_sep_delnot['word_' + str(i + 1)] != w]
                self.df_five_sep_delnot = self.df_five_sep_delnot.reset_index(drop=True)

    # スコアの計算
    def calc_score_easy(self, answord, predword):
        score = []
        for str_ans, str_pred in zip(answord, predword):
            if str_ans == str_pred:
                score.append(2)
            elif str_pred in answord:
                score.append(1)
            else:
                score.append(0)
        return score

    # ある入力単語が与えられたとき，スコアを全単語で計算
    def calc_all_score(self, word_input):
        scores = []
        for word in self.df_five_sep['word']:
            score = self.calc_score_easy(word_input, word)
            scores.append(tuple(score))
        return scores

    # エントロピーの計算
    def calc_entropy(self, word_input):
        scores = self.calc_all_score(word_input)
        entropy = 0
        for num in Counter(scores).values():
            entropy += -num / len(scores) * np.log2(num / len(scores))
        return entropy

    def calc_entropy_all(self):
        # 全単語のエントロピーを計算 (存在しない単語のみ省いたリストで計算)
        entropy_all = []
        for word in self.df_five_sep_delnot['word']:
            entropy_all.append((self.calc_entropy(word), word))

        entropy_all = sorted(entropy_all, reverse=True)
        print(entropy_all[0:10])


class PlayData(WordleData):
    """
    game class
    """
    def __init__(self, filename, gameiter = 5):
        super().__init__(filename)
        self.ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.answord = None
        self.gameiter = gameiter
        self.usr_decision = ['' for _ in range(5)]
        self.usr_notdecision = [set() for _ in range(5)]
        self.usr_included = set()
        self.usr_notindcluded = set()
        self.reset_alpha_list()

    def reset_alpha_list(self):
        """
        Resets the list of characters to be used in wordle.
        """
        self.alpha_list = list(self.ALPHA)
        self.used_list = [0 for i in range(len(self.ALPHA))]

    def calc_score(self, usr_ans):
        """
        calculate score
        """
        self.score = []
        i = 0
        for str_ans, str_usr in zip(self.answord, usr_ans):
            idx = self.ALPHA.index(str_usr)

            if str_ans == str_usr:  # green
                self.used_list[idx] = 2
                self.score.append(2)
                # add memory
                self.usr_decision[i] = str_usr
                self.usr_included.add(str_usr)
            elif str_usr in self.answord:  # yellow
                if self.used_list[idx] == 0:
                    self.used_list[idx] = 1
                self.score.append(1)
                # add memory
                self.usr_notdecision[i].add(str_usr)
                self.usr_included.add(str_usr)
            else:  # black
                self.used_list[idx] = -1
                self.score.append(0)
                # add memory
                self.usr_notdecision[i].add(str_usr)
                self.usr_notindcluded.add(str_usr)
            i += 1

    def print_state(self):
        """
        Prints the state of the game.
        """
        #print("Your score: ", self.score)
        #print("Your decision: ", self.usr_decision)
        #print("Your not-decision: ", self.usr_notdecision)
        #print("Your included: ", self.usr_included)
        #print("Your not-included: ", self.usr_notindcluded)
     
        self.cut_df()

    def get_input(self):
        """
        Get input from user.
        """
        while True:
            usr_ans = input("Enter your answer: ")
            usr_ans = usr_ans.upper()
       
            if self.serch_word(usr_ans) and len(usr_ans) == 5:
                break
            if usr_ans == "HELP":
                self.print_five_sep()
            else:
                print("Invalid input!")
        return usr_ans

    def get_input_wordle(self):
        """
        get input from wordle query
        """
        while True:
            usr_ans = input("Enter wordle query: ")
            if len(usr_ans) == 5:
                break
            else:
                print("Invalid input!")
        return usr_ans

    def cut_df(self):
        self.del_decision()
        self.del_notdecision()
        self.del_include()
        self.del_notinclude()
        print(self.df_five_sep.shape)

    def game(self):
        """
        Game function.
        """
        self.answord = self.choice_word()
        self.reset_alpha_list()
        for i in range(self.gameiter):
            usr_ans = self.get_input()
            usr_ans = usr_ans.upper()
            if usr_ans == self.answord:
                print("You win!")
                exit()
            else:
                self.calc_score(usr_ans)
                self.print_state()
        print("You lose!")

    def solver(self):
        """
        Solver function.
        """
        self.reset_alpha_list()
        for i in range(self.gameiter):
            usr_ans = self.get_input()
            score = self.get_input_wordle()
            # calc
            self.score = []
            for i in range(5):
                self.score.append(str(score[i]))
                str_usr = usr_ans[i]
                if score[i] == '2':  # 2: green
                    self.usr_decision[i] = str_usr
                    self.usr_included.add(str_usr)
                elif score[i] == '1':  # 1: yellow
                    self.usr_notdecision[i].add(str_usr)
                    self.usr_included.add(str_usr)
                else:  # 0: black
                    self.usr_notdecision[i].add(str_usr)
                    self.usr_notindcluded.add(str_usr)
            self.print_state()


def main():
    file_name = 'ejdic-hand-txt/ejdict-hand-utf8.txt'
    pg = PlayData(file_name, gameiter=10)
    pg.read_data_nltk()
    pg.df_sep()
    #pg.game()
    pg.solver()


if __name__ == '__main__':
    main()
