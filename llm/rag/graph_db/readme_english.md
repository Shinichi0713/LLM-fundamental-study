# 1. Fundamental Components

A graph database is primarily composed of the following three elements.

### Node

A **node** represents the data entity itself, such as a **person, object, or location**.
This is roughly equivalent to a **record in an RDB table**.

### Edge (Relationship)

An **edge** represents the relationship between nodes.
Edges typically have **directionality** and can contain semantic meaning such as:

* “is a friend of”
* “purchased”

Edges can also contain their own attributes.

### Property

A **property** is additional information attached to nodes or edges, such as:

* a person's name
* the purchase date of an item

---

# 2. Why Graph Databases Are Needed (Difference from RDB)

Traditional RDBs can represent relationships, but when relationships become complex, operations called **JOINs** become frequent and computationally expensive.

### In an RDB

If we attempt to query something like:

> “List the products purchased by the friends of A's friends.”

Multiple tables must be joined repeatedly. As the depth of the relationship increases, **performance deteriorates significantly**.

### In a Graph DB

Since nodes are already connected via **edges**, queries simply **traverse pointers** along those edges.
Even if the relationship depth becomes large, **performance remains relatively stable**.

---

# 3. Major Use Cases

Graph databases excel in domains where relationships between data are complex.

### Social Network Recommendations

They can instantly identify:

* “friends of friends”
* people with shared interests

### Fraud Detection (e.g., credit cards)

Connections between:

* payment terminals
* IP addresses
* bank accounts

can be analyzed to detect suspicious transaction patterns.

### Knowledge Graphs

Systems like search engines connect disparate information such as:

* people
* books
* locations
* historical facts

and derive answers based on contextual relationships.

### Supply Chain Management

Graph models can represent complex dependencies among:

* components
* factories
* logistics routes

This allows rapid identification of downstream impact when disruptions occur.

---

# 4. Representative Products

Some well-known graph database systems include:

* **Neo4j**
  The most widely used open-source graph database.

* **Amazon Neptune**
  A fully managed graph database service provided by AWS.

* **Memgraph** / **ArangoDB**
  Systems optimized for in-memory processing or hybrid data models.

---

# Experiment

If you want to experiment with a real graph query language such as **Cypher**, the following workflow is recommended.

---

# Step: Use Neo4j Aura (Free Cloud Version)

Neo4j provides a free cloud version that can be used simply by logging in.

[https://neo4j.com/cloud/aura-free/](https://neo4j.com/cloud/aura-free/)

### Steps

1. Create a free instance in Aura and obtain the **Connection URL, username, and password**.
2. Connect from Google Colab using the following code.

```python
!pip install neo4j pyvis

from neo4j import GraphDatabase

# Connection information
uri = "neo4j+s://xxxxxxx.databases.neo4j.io"
user = "neo4j"
password = "your_password"

driver = GraphDatabase.driver(uri, auth=(user, password))

def create_friends(tx):
    query = (
        "MERGE (a:Person {name: $name1}) "
        "MERGE (b:Person {name: $name2}) "
        "MERGE (a)-[:FRIEND]->(b)"
    )
    tx.run(query, name1=name1, name2=name2)

with driver.session() as session:
    session.execute_write(create_friends, "Alice", "Bob")

print("Data registration completed.")
driver.close()
```

When you create a Neo4j Aura instance, a file called **credentials.txt** is automatically downloaded in your browser.

Example:

```
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_generated_password
```

---

# Visualizing the Stored Data

You can confirm the stored graph using the following code.

```python
from neo4j import GraphDatabase
from pyvis.network import Network
import IPython

def get_graph_data(tx):
    query = "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50"
    return list(tx.run(query))

net = Network(notebook=True, height="500px", width="100%",
              bgcolor="#222222", font_color="white", cdn_resources='remote')

with driver.session() as session:
    results = session.execute_read(get_graph_data)

    for record in results:
        node_n = record["n"]
        node_m = record["m"]
        rel = record["r"]

        net.add_node(node_n.id,
                     label=node_n.get("name", str(list(node_n.labels))),
                     title=str(dict(node_n)))

        net.add_node(node_m.id,
                     label=node_m.get("name", str(list(node_m.labels))),
                     title=str(dict(node_m)))

        net.add_edge(node_n.id, node_m.id, label=rel.type)

net.show("graph.html")
IPython.display.HTML("graph.html")
```

This produces a network visualization of the stored relationships.


![1772890593028](image/README/1772890593028.png)

---

# Example Scenario

To demonstrate the real power of graph databases—**multi-layer relationships (n-degree separation)**—we construct a larger scenario.

---

# Story: “Tech Gadget Ecosystem”

### Entities

Users
Alice, Bob, Charlie, David, Eve

Items
iPhone 15, MacBook M3, Pixel 8, Galaxy S24, Sony WH-1000XM5

Brands
Apple, Google, Samsung, Sony

Categories
Smartphone, Laptop, Audio

Stores
Shinjuku Store, Shibuya Store

---

### Relationships

* Who purchased what (`PURCHASED`)
* Who is friends with whom (`FRIEND`)
* Product–brand relationships (`BRANDED_BY`)
* Product–store relationships (`STOCKED_IN`)

---

# 1. Register Large Story Data

```python
def create_massive_story(tx):
    tx.run("MATCH (n) DETACH DELETE n")

    tx.run("""
    CREATE (c1:Category {name: 'Smartphone'}),
           (c2:Category {name: 'Laptop'}),
           (c3:Category {name: 'Audio'}),
           (b1:Brand {name: 'Apple'}),
           (b2:Brand {name: 'Google'}),
           (b3:Brand {name: 'Samsung'}),
           (b4:Brand {name: 'Sony'})
    """)

    tx.run("""
    CREATE (i1:Item {name:'iPhone 15',price:120000}),
           (i2:Item {name:'MacBook M3',price:200000}),
           (i3:Item {name:'Pixel 8',price:90000}),
           (i4:Item {name:'Galaxy S24',price:110000}),
           (i5:Item {name:'WH-1000XM5',price:50000})
    """)
```

(This code continues to build relationships among users, items, and brands.)



# 2. Graph Visualization

Nodes are color-coded by type:

* User
* Item
* Brand
* Category

Using **PyVis**, the network structure becomes visually interpretable.

![1772890804095](image/README/1772890804095.png)

# 3. Advanced Recommendation Query

One powerful example query is:

> “Recommend items purchased by a friend's friend that the user has not yet bought.”

```python
MATCH (me:User {name:$name})-[:FRIEND*1..2]-(friend)
      -[:PURCHASED]->(item)-[:BRANDED_BY]->(brand)
WHERE NOT (me)-[:PURCHASED]->(item)
RETURN item.name, brand.name
```

Example output:

```
Recommendation: WH-1000XM5 (Sony)
Reason: Your friend Charlie owns it.
```

---

# Advantages of This Structure

### 1. Utilizing Sparse Data

Even users not directly connected to Alice (e.g., Charlie or David) can be reached via:

```
[:FRIEND*1..2]
```

In an RDB, this would require multiple **self-joins**, making the query far more complex.

---

### 2. Flexible Business Logic

New conditions such as:

* same brand preference
* same store visits

can be implemented simply by modifying **Cypher path patterns**.

---

### 3. Visual Insight

When visualized with PyVis, the network shows clusters such as:

* Apple ecosystem users
* Google ecosystem users

making social and product relationships immediately visible.

---

# Summary

Through this discussion we explored:

* Graph database fundamentals
* Practical implementation using Google Colab
* Advanced recommendation queries

The core reason graph databases are so powerful is:

> **Relationships themselves are stored as data.**

---

## 1. Essence of Graph Databases

In traditional RDB systems:

* Relationships are reconstructed using **JOINs at runtime**.

In graph databases:

* Relationships are stored directly as **edges**.

Thus traversal is extremely efficient.

---

## 2. Significance of Multi-Layer Modeling

In our example we connected three types of information:

* **Social relationships** (User–User)
* **Transactions** (User–Item)
* **Knowledge** (Item–Brand/Category)

This unified network enables queries such as:

> “Smartphones from the Apple brand purchased by a friend's friend of Alice.”

---

## 3. Tool Stack Roles

| Tool         | Role                      | Best Use                            |
| ------------ | ------------------------- | ----------------------------------- |
| **NetworkX** | In-memory graph analysis  | Research and algorithm experiments  |
| **Neo4j**    | Persistent graph database | Production-scale services           |
| **PyVis**    | Visualization             | Understanding complex relationships |

---

## 4. Why Graph Databases Are Gaining Attention

An important modern trend is integration with **LLM systems**.

### GraphRAG

**Retrieval-Augmented Generation** combined with graph databases allows LLMs to reference **structured knowledge graphs** instead of relying only on text retrieval.

This significantly reduces **hallucinations** and improves factual accuracy.

---

## 5. Future Directions

Graph databases open the door to further advanced capabilities:

* Graph algorithms such as **PageRank**
* Community detection (finding groups with shared interests)
* Temporal graphs using edge properties such as timestamps
* Large-scale graph processing on terabyte-level datasets in cloud infrastructure

---

Graph databases therefore represent a powerful paradigm for modeling **real-world systems where relationships matter as much as the data itself**.
