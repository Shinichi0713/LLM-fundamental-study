import yaml
import os

# book.yamlを読み込む
dir_parent = os.path.dirname(os.path.abspath(__file__))
with open(f'{dir_parent}/book.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

output_filename = config['merged-file-name']
md_files = config['md-files']

with open(os.path.join(dir_parent, output_filename), 'w', encoding='utf-8') as outfile:
    for i, md_file in enumerate(md_files):
        with open(os.path.join(dir_parent, md_file), 'r', encoding='utf-8') as infile:
            content = infile.read()
            outfile.write(content)
            # ファイル間に区切り行を入れる場合は以下を有効化
            # outfile.write('\n\n---\n\n')
            # 最後のファイル以外に改行を追加
            if i != len(md_files) - 1:
                outfile.write('\n\n')

print(f'Merged markdown saved as: {output_filename}')