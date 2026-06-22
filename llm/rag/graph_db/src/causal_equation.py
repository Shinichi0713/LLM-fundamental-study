import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel

# 1. データの生成 (SCMの実装)
np.random.seed(42)
num_samples = 1000

# 外生変数 (U): 観測できないノイズや個体差
u_temp = np.random.normal(0, 1, num_samples)
u_ice = np.random.normal(0, 1, num_samples)
u_acc = np.random.normal(0, 1, num_samples)

# 構造方程式 (Assignment)
# 気温 (Temp) は外生的な要因で決まる
temp = 20 + 5 * u_temp

# アイスの売上 (Ice) := f(Temp, U)
ice_sales = 2 * temp + 10 * u_ice

# 水難事故 (Accident) := f(Temp, U)
# ※アイスの売上は事故に直接影響を与えていないことに注目！
accidents = 0.5 * temp + 2 * u_acc

df = pd.DataFrame({
    'Temp': temp,
    'IceSales': ice_sales,
    'Accidents': accidents
})

# 2. 因果グラフの定義
# 矢印：Temp -> IceSales, Temp -> Accidents
causal_graph = """
digraph {
    Temp -> IceSales;
    Temp -> Accidents;
}
"""

model = CausalModel(
    data=df,
    treatment='IceSales',   # 原因と仮定したい変数
    outcome='Accidents',    # 結果と仮定したい変数
    graph=causal_graph
)

model.view_model() # グラフの可視化

print(f"相関係数 (IceSales vs Accidents): {df['IceSales'].corr(df['Accidents']):.3f}")
# 出力例: 0.90 以上の高い相関が出るはずです。

# 因果効果の特定 (バックドア基準などを用いて、どの変数を調整すべきか数学的に判断)
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# 因果効果の推定 (実際に do演算をシミュレート)
estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")

print(f"推定された因果効果 (IceSales -> Accidents): {estimate.value:.3f}")
# 出力例: ほぼ 0 になります。