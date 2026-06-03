from __future__ import annotations
from typing import Union

class Ordinal:
    def __init__(self, tag: str, arg: Union[int, list[Ordinal], None] = None):
        self.tag = tag  # "zero", "succ", "limit"
        self.arg = arg  # 後続なら前の順序数、極限ならより小さい順序数のリストなど

    @staticmethod
    def zero() -> Ordinal:
        return Ordinal("zero")

    @staticmethod
    def succ(prev: Ordinal) -> Ordinal:
        return Ordinal("succ", prev)

    @staticmethod
    def limit(smaller: list[Ordinal]) -> Ordinal:
        return Ordinal("limit", smaller)

    def __repr__(self):
        if self.tag == "zero":
            return "0"
        elif self.tag == "succ":
            return f"succ({self.arg})"
        elif self.tag == "limit":
            return f"lim[{','.join(map(str, self.arg))}]"

def F_V(alpha: Ordinal, f_dict: dict[Ordinal, set]) -> set:
    """V_alpha を定義する再帰的定義式の例"""
    if alpha.tag == "zero":
        return set()  # V_0 = ∅

    elif alpha.tag == "succ":
        beta = alpha.arg  # alpha = beta+1
        V_beta = f_dict[beta]
        # V_{beta+1} = P(V_beta) の有限近似として「V_beta の有限部分集合全体」
        # 実際には有限集合しか扱えないので、冪集合全体は無理だがイメージとして
        return {frozenset(subset) for subset in powerset(V_beta)}

    elif alpha.tag == "limit":
        smaller = alpha.arg  # alpha より小さい順序数のリスト
        # V_alpha = ∪_{beta < alpha} V_beta
        union_set = set()
        for beta in smaller:
            union_set.update(f_dict[beta])
        return union_set

def powerset(s):
    """有限集合 s の冪集合（有限部分集合全体）を返す"""
    from itertools import chain, combinations
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def build_transfinite(F, ordinals: list[Ordinal]) -> dict[Ordinal, object]:
    """ordinals に列挙された順序数に対して F を用いて f を構成する"""
    f_dict = {}
    for alpha in ordinals:
        # f|_alpha は既に f_dict に格納されているとみなす
        f_dict[alpha] = F(alpha, f_dict)
    return f_dict

def F_add(beta: Ordinal, f_dict: dict[Ordinal, Ordinal]) -> Ordinal:
    """α+beta を定義する再帰的定義式（αは固定）"""
    if beta.tag == "zero":
        return alpha  # α+0 = α

    elif beta.tag == "succ":
        gamma = beta.arg  # beta = gamma+1
        alpha_plus_gamma = f_dict[gamma]
        return Ordinal.succ(alpha_plus_gamma)  # α+(gamma+1) = (α+gamma)+1

    elif beta.tag == "limit":
        smaller = beta.arg
        # α+beta = sup{α+gamma | gamma < beta}
        # 有限近似として、smaller の中で最大のものを取るなど
        # ここでは単にリストの最後の要素を採用（単純化）
        return smaller[-1]

def define_addition(alpha: Ordinal, betas: list[Ordinal]) -> dict[Ordinal, Ordinal]:
    """α+beta を beta について超限帰納的に定義"""
    f_dict = {}
    for beta in betas:
        f_dict[beta] = F_add(beta, f_dict)
    return f_dict
