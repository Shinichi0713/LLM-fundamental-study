import re

def add_space_around_dollar_near_symbols(text: str) -> str:
    """
    Markdownテキスト中のインライン数式について、
    - 記号（ここでは \\{, \\}, \\dots, , など）の前後に来る $ の外側に半角スペースを挿入
    - ただし、インライン数式の内部（$...$ の間）にはスペースを追加しない
    """

    # 1. インライン数式全体をマスク（一時的に別の文字列に置換）
    inline_math_pattern = r'\$[^$]+\$'  # 単純なインライン数式用（入れ子なしを仮定）
    placeholders = []

    def mask_math(match):
        """マッチしたインライン数式をプレースホルダに置換"""
        placeholders.append(match.group(0))
        return f"@@MATH_{len(placeholders)-1}@@"

    masked_text = re.sub(inline_math_pattern, mask_math, text)

    # 2. マスク済みテキストに対して、記号前後の $ の外側にスペースを挿入
    # ここでは例として \{, \}, \dots, , を「記号」とみなす

    # 2.1 記号の前に来る $ の左側にスペースを追加
    # パターン: $ の直後に空白0個以上＋記号が続く場合
    pattern_before_symbol = r'(?<!\s)\$(?=\s*(\\\{|\\\}|\\dots|,))'
    masked_text = re.sub(pattern_before_symbol, r' $', masked_text)

    # 2.2 記号の後に来る $ の右側にスペースを追加
    # パターン: 記号＋空白0個以上＋$ が続き、その直後に空白がない場合
    pattern_after_symbol = r'((\\\{|\\\}|\\dots|,))(?:\s*)\$(?!\s)'
    masked_text = re.sub(pattern_after_symbol, r'\1 $', masked_text)

    # 3. マスクを解除（プレースホルダを元のインライン数式に戻す）
    for i, orig_math in enumerate(placeholders):
        masked_text = masked_text.replace(f"@@MATH_{i}@@", orig_math)

    return masked_text


# 使用例
if __name__ == "__main__":
    sample_text = r"""
例1：$\{1,2,3,\dots\}$
例2：$\{x \mid x>0\}$ は正の実数全体の集合です。
例3：$a,b,c$ のようにカンマ区切りでも使います。
    """

    result = add_space_around_dollar_near_symbols(sample_text)
    print("変換前:")
    print(sample_text)
    print("\n変換後:")
    print(result)