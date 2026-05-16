import re
import os

def convert_to_note_tex(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} が見つかりません。")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. ディスプレイ形式 \[ ... \] を $$ ... $$ に変換
    # フラグ re.DOTALL で複数行にわたる数式にも対応
    content = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', content, flags=re.DOTALL)

    # 2. インライン形式 \( ... \) を $$ ... $$ に変換
    # noteで目立たせたいとのことですので、こちらも $$ に統一します
    content = re.sub(r'\\\((.*?)\\\)', r'$$\1$$', content)

    # 3. 既存の単一 $ ... $ (インライン) もすべて $$ ... $$ に置き換えたい場合
    # ※数式以外の $ 記号に反応しないよう、前後にスペースや改行があるケースを想定
    # 不要な場合は以下の1行をコメントアウトしてください
    content = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', r'$$\1$$', content)

    # 結果を書き出し
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    return content

def format_to_note_style(content, output_file):

    # 1. 各種Latex形式（\[ \], \( \), $$ $$）を統一して抽出・整形する関数
    def clean_and_wrap(match):
        # マッチした数式の中身を取り出す（グループ1）
        formula = match.group(1)
        # 改行をスペースに置換し、前後の空白を削除
        cleaned_formula = " ".join(formula.split())
        # $${ 数式 }$$ の形式で返す
        return f"$${{ {cleaned_formula} }}$$"

    # --- 置換処理 ---
    
    # パターンA: \[ ... \] 形式（複数行対応）
    content = re.sub(r'\\\[(.*?)\\\]', clean_and_wrap, content, flags=re.DOTALL)
    
    # パターンB: \( ... \) 形式
    content = re.sub(r'\\\((.*?)\\\)', clean_and_wrap, content)
    
    # パターンC: 既存の $$ ... $$ 形式（複数行対応）
    # ※既に $$ で囲まれているものも $${ }$$ に統一
    content = re.sub(r'\$\$(.*?)\$\$', clean_and_wrap, content, flags=re.DOTALL)
    
    # パターンD: 単一の $ ... $ 形式
    content = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', clean_and_wrap, content)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"変換が完了しました！\n出力先: {output_file}")

# 実行設定
input_md = r"D:\PycharmProjects\LLM-research\LLM-fundamental-study\llm\doc\13_Towards_Monosemanticity.md"   # 変換元のファイル名
output_md = r"D:\PycharmProjects\LLM-research\LLM-fundamental-study\llm\doc\13_Towards_Monosemanticity.md" # 変換後のファイル名

content = convert_to_note_tex(input_md, output_md)
format_to_note_style(content, output_md)