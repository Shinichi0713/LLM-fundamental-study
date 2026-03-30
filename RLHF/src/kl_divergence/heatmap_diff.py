import numpy as np
import matplotlib.pyplot as plt

def kl_divergence(p, q, eps=1e-8):
    """
    離散分布 P, Q に対する KL(P || Q) を計算
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    p = p + eps
    q = q + eps
    
    p = p / p.sum()
    q = q / q.sum()
    
    return np.sum(p * np.log(p / q))

# P を [0.7, 0.3] に固定
p_fixed = np.array([0.7, 0.3])

q1_vals = np.linspace(0.01, 0.99, 50)
q2_vals = np.linspace(0.01, 0.99, 50)

kl_pq_grid = np.zeros((len(q1_vals), len(q2_vals)))
kl_qp_grid = np.zeros((len(q1_vals), len(q2_vals)))

for i, q1 in enumerate(q1_vals):
    for j, q2 in enumerate(q2_vals):
        q = np.array([q1, q2])
        q = q / q.sum()
        
        kl_pq = kl_divergence(p_fixed, q)
        kl_qp = kl_divergence(q, p_fixed)
        
        kl_pq_grid[i, j] = kl_pq
        kl_qp_grid[i, j] = kl_qp

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(kl_pq_grid, extent=[0.01, 0.99, 0.01, 0.99], origin='lower', cmap='hot')
axes[0].set_xlabel('q1')
axes[0].set_ylabel('q2')
axes[0].set_title('KL(P_fixed || Q)')
axes[0].plot(0.7, 0.3, 'ro', markersize=8, label='P_fixed = [0.7, 0.3]')
axes[0].legend()
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(kl_qp_grid, extent=[0.01, 0.99, 0.01, 0.99], origin='lower', cmap='hot')
axes[1].set_xlabel('q1')
axes[1].set_ylabel('q2')
axes[1].set_title('KL(Q || P_fixed)')
axes[1].plot(0.7, 0.3, 'ro', markersize=8, label='P_fixed = [0.7, 0.3]')
axes[1].legend()
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()