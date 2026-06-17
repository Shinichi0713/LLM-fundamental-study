import re
from pathlib import Path

def add_space_around_dollar_near_symbols(text: str) -> str:
    """
    Markdownテキスト中のインライン数式について、
    - 記号（\\{, \\}, \\dots, , や \\leq, \\preceq, \\iff など）の前後に来る $ の外側に半角スペースを挿入
    - ただし、インライン数式の内部（$...$ の間）にはスペースを追加しない
    """
    # 1. インライン数式全体をマスク
    inline_math_pattern = r'\$[^$]+\$'  # 単純なインライン数式用（入れ子なしを仮定）
    placeholders = []

    def mask_math(match):
        placeholders.append(match.group(0))
        return f"@@MATH_{len(placeholders)-1}@@"

    masked_text = re.sub(inline_math_pattern, mask_math, text)

    # 2. マスク済みテキストに対して、記号前後の $ の外側にスペースを挿入
    # 記号の例：\{, \}, \dots, , に加えて \leq, \preceq, \iff など
    symbols = [
        r'\\\{', r'\\\}', r'\\dots', ',',   # 元からあったもの
        r'\\leq', r'\\preceq', r'\\iff',    # 追加：よく使う関係演算子
        r'\\geq', r'\\sim', r'\\simeq',     # 必要に応じてさらに追加
        r'\\equiv', r'\\approx', r'\\propto',
    ]
    symbols_pattern = '|'.join(symbols)

    # 2.1 記号の前に来る $ の左側にスペースを追加
    pattern_before_symbol = rf'(?<!\s)\$(?=\s*({symbols_pattern}))'
    masked_text = re.sub(pattern_before_symbol, r' $', masked_text)

    # 2.2 記号の後に来る $ の右側にスペースを追加
    pattern_after_symbol = rf'(({symbols_pattern}))(?:\s*)\$(?!\s)'
    masked_text = re.sub(pattern_after_symbol, r'\1 $', masked_text)

    # 3. マスクを解除
    for i, orig_math in enumerate(placeholders):
        masked_text = masked_text.replace(f"@@MATH_{i}@@", orig_math)

    return masked_text


def process_markdown_file(input_path: str, output_path: str) -> None:
    """
    Markdownファイルを読み込み、数式周りのスペースを整形して別ファイルに保存する
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"エラー: 入力ファイル '{input_path}' が見つかりません。")
        return

    with input_file.open("r", encoding="utf-8") as f:
        content = f.read()

    new_content = add_space_around_dollar_near_symbols(content)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"変換完了: '{input_path}' -> '{output_path}'")


if __name__ == "__main__":
    # ここで入力ファイル名と出力ファイル名を直接指定
    input_md = r"D:\PycharmProjects\LLM-research\LLM-fundamental-study\basics_nn\sets_theory\doc\6_order_set_out.md"
    output_md = r"D:\PycharmProjects\LLM-research\LLM-fundamental-study\basics_nn\sets_theory\doc\6_order_set_out.md"

    process_markdown_file(input_md, output_md)