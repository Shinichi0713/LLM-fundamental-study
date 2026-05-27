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
            ci, cj = i + pad, j + pad
            inner = anomaly_padded[ci - inner_size//2 : ci + inner_size//2 + 1,
                                   cj - inner_size//2 : cj + inner_size//2 + 1]
            inner_mean = np.mean(inner)
            
            outer = anomaly_padded[ci - outer_size//2 : ci + outer_size//2 + 1,
                                  cj - outer_size//2 : cj + outer_size//2 + 1]
            mask = np.ones(outer.shape, dtype=bool)
            mask[inner_size//2 : -inner_size//2, inner_size//2 : -inner_size//2] = False
            outer_mean = np.mean(outer[mask])
            
            isolation_map[i, j] = inner_mean - outer_mean
    
    return isolation_map

def anomaly_map_to_bboxes_with_blob_score(
    anomaly_map,
    alpha=0.5, beta=0.3, gamma=0.2,
    thresh=0.5, min_area=10
):
    """
    アノマリマップからBBOXを抽出し、塊らしさも考慮したスコアを付与
    """
    # 孤立度マップ計算
    isolation_map = compute_isolation_map(anomaly_map)
    
    # 正規化
    anomaly_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    isolation_norm = (isolation_map - isolation_map.min()) / (isolation_map.max() - isolation_map.min() + 1e-8)
    
    # 統合スコア（画素レベル）
    score_map = alpha * anomaly_norm + beta * isolation_norm
    
    # 閾値処理
    mask = (score_map > thresh).astype(np.uint8)
    
    # 連結成分解析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    bboxes = []
    for i in range(1, num_labels):  # 0は背景
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        
        # BBOX内の異常度・孤立度の代表値（例: 最大値）
        region_anomaly = np.max(anomaly_map[y:y+h, x:x+w])
        region_isolation = np.max(isolation_map[y:y+h, x:x+w])
        
        # 塊らしさの指標
        # 1) コンパクトネス（周長が必要なので輪郭抽出）
        contour_mask = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        perimeter = cv2.arcLength(contours[0], True)
        compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-8)
        
        # 2) Fill Ratio
        fill_ratio = area / (w * h)
        
        # 塊らしさスコア（例: compactness と fill_ratio の平均）
        blob_score = 0.5 * compactness + 0.5 * fill_ratio
        
        # 正規化（必要に応じて）
        # ここでは簡略化のため、0〜1の範囲になることを前提
        
        # 最終スコア（例: 線形結合）
        final_score = (
            alpha * region_anomaly +
            beta * region_isolation +
            gamma * blob_score
        )
        
        bboxes.append((x, y, w, h, final_score, blob_score))
    
    # スコア順にソート
    bboxes.sort(key=lambda b: b[4], reverse=True)
    return bboxes