from neo4j import GraphDatabase

# 1. 接続情報の設定
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "your-password")

def extract_graph_schema():
    print("🌐 GraphDBから現在の登録スキーマ（設計図）を抽出しています...\n")
    
    # スキーマ構造を格納する辞書
    schema_relations = set()
    node_labels = set()

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            
            # クエリ1: 実際に存在するすべての (ソースラベル)-[:関係性]->(ターゲットラベル) の組み合わせを取得
            cypher_schema = """
            MATCH (s)-[r]->(t)
            RETURN DISTINCT labels(s)[0] AS source, type(r) AS relation, labels(t)[0] AS target
            """
            records = session.run(cypher_schema)
            for record in records:
                if record["source"] and record["target"]:
                    schema_relations.add((record["source"], record["relation"], record["target"]))
                    node_labels.add(record["source"])
                    node_labels.add(record["target"])
            
            # クエリ2: 孤立しているノード（関係性はないがラベルだけ存在するもの）の取得
            cypher_isolated = """
            MATCH (n) 
            WHERE NOT (n)-[]-()
            RETURN DISTINCT labels(n)[0] AS isolated_label
            """
            isolated_records = session.run(cypher_isolated)
            for record in isolated_records:
                if record["isolated_label"]:
                    node_labels.add(record["isolated_label"])

    # 2. 結果を綺麗に整形して出力
    print("==================================================")
    print("      📊 現在のGRAPH DB スキーマ確認レポート       ")
    print("==================================================")
    
    print("\n[1] 登録されているノードの型（ラベル一覧）:")
    if node_labels:
        for label in sorted(node_labels):
            print(f"  - ● {label}")
    else:
        print("  (ノードは登録されていません)")

    print("\n[2] 許可/登録されているトリプル（関係性のルール）:")
    if schema_relations:
        # 見やすいように主語ごとにソートして表示
        for src, rel, tgt in sorted(schema_relations):
            print(f"  - ({src})  ───[ {rel} ]───>  ({tgt})")
    else:
        print("  (関係性は登録されていません)")
        
    print("\n==================================================")

if __name__ == "__main__":
    try:
        extract_graph_schema()
    except Exception as e:
        print(f"エラー: データベースの解析に失敗しました。\n{e}")