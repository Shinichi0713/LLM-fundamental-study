import re
import sys
from pathlib import Path

def add_space_around_dollar_near_symbols(text: str) -> str:
    """
    Markdownテキスト中のインライン数式について、
    - 記号（ここでは \\{, \\}, \\dots, , など）の前後に来る $ の外側に半角スペースを挿入
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
    # ここでは例として \{, \}, \dots, , を「記号」とみなす

    # 2.1 記号の前に来る $ の左側にスペースを追加
    pattern_before_symbol = r'(?<!\s)\$(?=\s*(\\\{|\\\}|\\dots|,))'
    masked_text = re.sub(pattern_before_symbol, r' $', masked_text)

    # 2.2 記号の後に来る $ の右側にスペースを追加
    pattern_after_symbol = r'((\\\{|\\\}|\\dots|,))(?:\s*)\$(?!\s)'
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
        sys.exit(1)

    # 入力ファイルを読み込み
    with input_file.open("r", encoding="utf-8") as f:
        content = f.read()

    # 変換処理
    new_content = add_space_around_dollar_near_symbols(content)

    # 出力ファイルに書き込み
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"変換完了: '{input_path}' -> '{output_path}'")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使い方: python script.py input.md output.md")
        sys.exit(1)

    input_md = sys.argv[1]
    output_md = sys.argv[2]

    process_markdown_file(input_md, output_md)