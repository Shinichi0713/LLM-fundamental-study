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


def powerset(s):
    """
    集合 s のべき集合（すべての部分集合の集合）を返す。
    """
    from itertools import chain, combinations
    # 0要素からlen(s)要素までのすべての組み合わせを生成
    return list(chain.from_iterable(
        combinations(s, r) for r in range(len(s) + 1)
    ))

def visualize_powerset(s):
    """
    集合 s のべき集合を整形して表示する。
    """
    ps = powerset(s)
    print(f"元の集合: {set(s)}")
    print(f"べき集合（要素数 {len(ps)}）:")
    for i, subset in enumerate(ps):
        # 空集合は ∅ で表示
        if not subset:
            print(f"  {i+1:2d}. ∅")
        else:
            print(f"  {i+1:2d}. {{{', '.join(map(str, subset))}}}")

# 例: {1, 2, 3} のべき集合を可視化
if __name__ == "__main__":
    S = [1, 2, 3]
    visualize_powerset(S)