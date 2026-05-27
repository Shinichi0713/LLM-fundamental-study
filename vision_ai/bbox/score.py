import cv2
import numpy as np

def compute_isolation_map(anomaly_map, inner_size=3, outer_size=7):
    """
    局所コントラストを利用した孤立度マップを計算
    """
    h, w = anomaly_map.shape
    pad = outer_size // 2
    anomaly_padded = np.pad(anomaly_map, pad, mode='reflect')
    
    isolation_map = np.zeros_like(anomaly_map)
    
    for i in range(h):
        for j in range(w):
            # 中心ウィンドウ
            ci, cj = i + pad, j + pad
            inner = anomaly_padded[ci - inner_size//2 : ci + inner_size//2 + 1,
                                   cj - inner_size//2 : cj + inner_size//2 + 1]
            inner_mean = np.mean(inner)
            
            # 外側リング
            outer = anomaly_padded[ci - outer_size//2 : ci + outer_size//2 + 1,
                                   cj - outer_size//2 : cj + outer_size//2 + 1]
            mask = np.ones(outer.shape, dtype=bool)
            mask[inner_size//2 : -inner_size//2, inner_size//2 : -inner_size//2] = False
            outer_mean = np.mean(outer[mask])
            
            # コントラストスコア
            isolation_map[i, j] = inner_mean - outer_mean
    
    return isolation_map

def anomaly_map_to_bboxes(anomaly_map, alpha=0.7, beta=0.3, thresh=0.5, min_area=10):
    """
    アノマリマップからBBOXを抽出
    """
    # 孤立度マップ計算
    isolation_map = compute_isolation_map(anomaly_map)
    
    # 正規化
    anomaly_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    isolation_norm = (isolation_map - isolation_map.min()) / (isolation_map.max() - isolation_map.min() + 1e-8)
    
    # 統合スコア
    score_map = alpha * anomaly_norm + beta * isolation_norm
    
    # 閾値処理
    mask = (score_map > thresh).astype(np.uint8)
    
    # 連結成分解析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    bboxes = []
    for i in range(1, num_labels):  # 0は背景
        x, y, w, h, area = stats[i]
        if area >= min_area:
            bboxes.append((x, y, w, h, np.max(score_map[y:y+h, x:x+w])))
    
    # スコア順にソート
    bboxes.sort(key=lambda b: b[4], reverse=True)
    return bboxes