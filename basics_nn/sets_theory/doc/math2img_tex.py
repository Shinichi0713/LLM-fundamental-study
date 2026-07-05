#!/usr/bin/env python3
import re
import os
import hashlib
import argparse
import tempfile
import shutil
import subprocess

def latex_to_png(latex_code, dpi=120, img_dir="math_images", display_mode=True):
    os.makedirs(img_dir, exist_ok=True)
    # モードをハッシュに含める（インラインとディスプレイで別画像）
    mode_flag = "display" if display_mode else "inline"
    latex_md5 = hashlib.md5((latex_code + mode_flag).encode("utf-8")).hexdigest()
    png_filename = f"{latex_md5}.png"
    png_path = os.path.join(img_dir, png_filename)

    if os.path.exists(png_path):
        with open(png_path, "rb") as f:
            return f.read()

    tmpdir = tempfile.mkdtemp()
    try:
        tmpdir_full = os.path.realpath(tmpdir)
        tex_path = os.path.join(tmpdir_full, "math.tex")
        pdf_path = os.path.join(tmpdir_full, "math.pdf")

        # インライン用は \textstyle、ディスプレイ用は \displaystyle
        if display_mode:
            math_env = r"\begin{displaymath}" + latex_code + r"\end{displaymath}"
        else:
            math_env = r"$\displaystyle " + latex_code + r"$"  # インラインでも大きめに

        with open(tex_path, "w", encoding="utf-8") as f:
            # luatexjaパッケージを読み込み、standaloneクラスで数式ぴったりにトリミングする
            f.write(r"""
\documentclass[preview,varwidth,margin=2pt]{standalone}
\usepackage{luatexja}  % 日本語対応
\usepackage{amsmath}
\usepackage{amssymb}
\pagestyle{empty}
\begin{document}
""" + math_env + r"""
\end{document}
""")

        # ファイルが正しく作成されたか確認
        tex_exists = os.path.exists(tex_path)
        print(f"[DEBUG] tex_path = {tex_path}")
        print(f"[DEBUG] os.path.exists(tex_path) = {tex_exists}")
        if tex_exists:
            with open(tex_path, "r", encoding="utf-8") as f:
                content = f.read()
                print(f"[DEBUG] math.tex 内容（先頭100文字）:\n{content[:100]}")

        # --- 修正箇所2: コンパイルコマンド（60行目〜70行目付近） ---
        # "pdflatex" から "lualatex" に変更
        result_tex = subprocess.run(
            ["lualatex", "-interaction=nonstopmode", tex_path],
            cwd=tmpdir_full,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )

        # ログ保存
        log_path = os.path.join(tmpdir_full, "math.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== stdout ===\n")
            f.write(result_tex.stdout)
            f.write("\n=== stderr ===\n")
            f.write(result_tex.stderr)

        # PDF存在確認
        pdf_exists = os.path.exists(pdf_path)
        print(f"[DEBUG] pdf_exists = {pdf_exists}")
        if pdf_exists:
            print("[DEBUG] LaTeXコンパイルは成功（PDFあり）")
        else:
            print("[DEBUG] LaTeXコンパイル失敗（PDFなし）")

        if not pdf_exists:
            print("=== LaTeXコンパイル失敗（詳細ログ） ===")
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                print(f.read())
            raise subprocess.CalledProcessError(
                result_tex.returncode, result_tex.args,
                output=result_tex.stdout, stderr=result_tex.stderr
            )

        # PDF→PNG変換（ImageMagick）
        magick_log_path = os.path.join(tmpdir_full, "magick.log")
        with open(magick_log_path, "w") as magick_log_file:
            # 修正後：-trim と +repage を追加して余白を完全に削ぎ落とす
            result_magick = subprocess.run(
                # ["magick", "-density", str(dpi), pdf_path, "-trim", "+repage", png_path],
                ["magick", "-density", str(dpi), pdf_path, png_path],
                stdout=magick_log_file,
                stderr=magick_log_file
            )
        if result_magick.returncode != 0:
            print("=== ImageMagick変換失敗 ===")
            with open(magick_log_path, "r", encoding="utf-8", errors="ignore") as f:
                print(f.read())
            raise subprocess.CalledProcessError(
                result_magick.returncode, result_magick.args
            )

        with open(png_path, "rb") as f:
            return f.read()

    except subprocess.CalledProcessError as e:
        print(f"LaTeX/ImageMagick処理でエラー: {e}")
        raise
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def replace_math_with_images(md_text, dpi=120, img_dir="math_images"):
    # ディスプレイ数式: $$...$$
    display_pattern = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)
    # インライン数式: $...$
    inline_pattern = re.compile(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)')

    def replacer_display(match):
        latex_code = match.group(1).strip()
        png_path = latex_to_png(latex_code, dpi=dpi, img_dir=img_dir, display_mode=True)
        latex_md5 = hashlib.md5((latex_code + "display").encode("utf-8")).hexdigest()
        png_filename = f"{latex_md5}.png"
        path = os.path.join(img_dir, png_filename).replace("\\", "/")
        
        # ★ class="math-display" を付与
        return f'\n\n<div class="math-display-container"><img src="{path}" class="math-display" /></div>\n\n'

    def replacer_inline(match):
        latex_code = match.group(1).strip()
        png_path = latex_to_png(latex_code, dpi=dpi, img_dir=img_dir, display_mode=False)
        latex_md5 = hashlib.md5((latex_code + "inline").encode("utf-8")).hexdigest()
        png_filename = f"{latex_md5}.png"
        path = os.path.join(img_dir, png_filename).replace("\\", "/")
        
        # ★ class="math-inline" を付与。HTML直書きのstyleはKindleにバグを起こすため一度すべて排除
        return f'<img src="{path}" class="math-inline" />'

    md_text = display_pattern.sub(replacer_display, md_text)
    md_text = inline_pattern.sub(replacer_inline, md_text)
    return md_text

def main():
    parser = argparse.ArgumentParser(description="Markdownの数式をPNG画像に変換")
    parser.add_argument("input_file", help="入力Markdownファイル")
    parser.add_argument("--dpi", type=int, default=120, help="PNGのDPI（解像度）")
    parser.add_argument("--img-dir", default="math_images", help="画像出力ディレクトリ")
    parser.add_argument("-o", "--output", help="出力ファイル（省略時は input_file.editted.md）")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        md_text = f.read()

    new_text = replace_math_with_images(md_text, dpi=args.dpi, img_dir=args.img_dir)

    out_path = args.output or args.input_file.replace(".md", ".editted.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(new_text)

    print(f"変換完了: {out_path}")

if __name__ == "__main__":
    main()

# python math2img_tex.py D:\PycharmProjects\LLM-research\LLM-fundamental-study\basics_nn\sets_theory\doc\3_indexed_set.md --dpi 150 --img-dir tmp -o D:\PycharmProjects\LLM-research\LLM-fundamental-study\basics_nn\sets_theory\doc\3_indexed_set_converted.md
# python math2img_tex.py D:\PycharmProjects\LLM-research\LLM-fundamental-study\basics_nn\sets_theory\doc\merged_output.md --dpi 150 --img-dir tmp -o D:\PycharmProjects\LLM-research\LLM-fundamental-study\basics_nn\sets_theory\doc\merged_output_converted.md