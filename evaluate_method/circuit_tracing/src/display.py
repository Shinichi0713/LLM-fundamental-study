from sklearn.linear_model import LinearRegression
import networkx as nx
import matplotlib.pyplot as plt

edges = []

layer_names = list(top_neurons.keys())

for i in range(len(layer_names) - 1):
    l1 = layer_names[i]
    l2 = layer_names[i+1]
    
    act1 = activations[l1][0].detach().cpu().numpy()
    act2 = activations[l2][0].detach().cpu().numpy()
    
    for n2 in top_neurons[l2]:
        y = act2[:, n2]
        
        for n1 in top_neurons[l1]:
            x = act1[:, n1].reshape(-1, 1)
            
            reg = LinearRegression().fit(x, y)
            score = abs(reg.coef_[0])
            
            if score > 0.1:
                edges.append((f"{l1}:{n1}", f"{l2}:{n2}", score))


G = nx.DiGraph()

for src, dst, w in edges:
    G.add_edge(src, dst, weight=w)

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.5)

nx.draw(G, pos, with_labels=True, node_size=500, font_size=6)
plt.title("Pseudo Circuit Graph")
plt.show()