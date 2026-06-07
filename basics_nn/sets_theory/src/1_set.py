A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

print("A =", A)
print("B =", B)

# 和集合
print("A ∪ B =", A | B)          # または A.union(B)

# 共通部分
print("A ∩ B =", A & B)          # または A.intersection(B)

# 差集合
print("A \\ B =", A - B)         # または A.difference(B)
print("B \\ A =", B - A)

# 対称差（どちらか一方にだけ属する元）
print("A △ B =", A ^ B)          # または A.symmetric_difference(B)


from matplotlib_venn import venn2
import matplotlib.pyplot as plt

A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

venn2([A, B], set_labels=('A', 'B'))
plt.show()

from matplotlib_venn import venn3
import matplotlib.pyplot as plt

A = {1, 2, 3, 4}
B = {3, 4, 5, 6}
C = {4, 5, 7, 8}

venn3([A, B, C], set_labels=('A', 'B', 'C'))
plt.show()