import itertools
from collections import Counter
import nltk


alpha = "abcdefghijklmnopqrstuvwxyz"
# LS_five = set(list("".join(c) for c in itertools.permutations(alpha, 5)))
LS_five = set(list("".join(c) for c in itertools.product(alpha, repeat=5)))

nltk.download("brown")
nltk.download("words")

print(len(LS_five))
LS_cp = nltk.corpus.brown.words() + nltk.corpus.words.words()
C_cp = Counter(LS_cp)

Ans = []
for s in C_cp.keys():
    if s in LS_five and len(s) == 5:
        Ans.append(s)

Ans.sort()

with open("wordle_dict_all.txt", "w") as f:
    f.write("\n".join(Ans))
