# %%  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from collections import Counter

# %%　データの読み込み
df_five = pd.read_table("wordle_dict_all.txt", sep='\t', header=None, names=['word'])
df_five['word'] = df_five['word'].str.upper()

df_five_sep = df_five.copy()
df_five_sep['word_1'] = df_five_sep['word'].str[0]
df_five_sep['word_2'] = df_five_sep['word'].str[1]
df_five_sep['word_3'] = df_five_sep['word'].str[2]
df_five_sep['word_4'] = df_five_sep['word'].str[3]
df_five_sep['word_5'] = df_five_sep['word'].str[4]

df_five_sep
# %% 文字の出現頻度を比較
fig = plt.figure(figsize=(10, 16))

for i, col in enumerate(['word_1', 'word_2', 'word_3', 'word_4', 'word_5']):
    freq_w = pd.Series(df_five_sep[col].values.ravel()).value_counts().to_dict()
    ax = fig.add_subplot(5, 1, i + 1)
    ax.bar(freq_w.keys(), freq_w.values())
    ax.set_title(col)

# best word is "CARES"
# %% 


# スコアの計算
def calc_score(answord, predword):
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
def calc_all_score(word_input):
    scores = []
    for word in df_five['word']:
        score = calc_score(word_input, word)
        scores.append(tuple(score))
    return scores


# エントロピーの計算
def calc_entropy(word_input):
    scores = calc_all_score(word_input)
    entropy = 0
    for num in Counter(scores).values():
        entropy += -num / len(scores) * np.log2(num / len(scores))
    return entropy


calc_entropy('CARES')
# %% 全単語のエントロピーを計算
entropy_all = []
for word in df_five['word']:
    entropy_all.append((calc_entropy(word), word))

entropy_all = sorted(entropy_all, reverse=True)
entropy_all

# best word is "TARES" (entropy = 6.347), "CARES" (entropy = 6.341)
