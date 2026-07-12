import os
import json
import urllib.request

def download_github_folder(repo_url, output_dir="downloaded_folder"):
    """
    GitHubの特定のフォルダURLから中身を一括ダウンロードする関数
    例: https://github.com/ユーザー名/リポジトリ名/tree/メインブランチ/フォルダパス
    """
    # URLを解析してAPI用のパスに変換
    # 期待する入力: https://github.com/owner/repo/tree/branch/path/to/folder
    parts = repo_url.strip("/").split("/")
    if "tree" not in parts:
        print("エラー: URLに '/tree/ブランチ名/' が含まれていません。フォルダのURLを指定してください。")
        return

    idx = parts.index("tree")
    owner = parts[2]
    repo = parts[3]
    branch = parts[idx + 1]
    folder_path = "/".join(parts[idx + 2:])

    # GitHub API のエンドポイント
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{folder_path}?ref={branch}"
    
    # APIリクエスト（レートリミットに引っかかる場合はヘッダーにトークンを追加してください）
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    
    try:
        with urllib.request.urlopen(req) as response:
            items = json.loads(response.read().decode())
    except Exception as e:
        print(f"APIの取得に失敗しました: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)

    for item in items:
        if item["type"] == "file":
            file_name = item["name"]
            download_url = item["download_url"]
            file_path = os.path.join(output_dir, file_name)
            
            print(f"ダウンロード中: {file_name} ...")
            urllib.request.urlretrieve(download_url, file_path)
            
        elif item["type"] == "dir":
            # サブフォルダがある場合は再帰的に処理
            sub_folder_url = f"https://github.com/{owner}/{repo}/tree/{branch}/{item['path']}"
            sub_output_dir = os.path.join(output_dir, item["name"])
            download_github_folder(sub_folder_url, sub_output_dir)

    print("\nすべてのファイルのダウンロードが完了しました！")

# --- 使い方 ---
if __name__ == "__main__":
    # ここに対象のGitHubフォルダのURLを入力してください
    target_url = "https://github.com/owner/repo/tree/main/path/to/folder"
    
    download_github_folder(target_url, output_dir="my_github_files")