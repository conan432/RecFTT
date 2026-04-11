import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ================= 配置部分 =================
# 假设你的文件名是 item_meta.txt，分隔符是制表符 \t (根据你复制的样本推断)
# 如果是逗号分隔，请改 sep=','
FILE_PATH = 'item_meta.csv' 

# 你列出的所有类别列名
category_cols = [
    'i_Entertainment_c', 'i_Beauty_c', 'i_Home Services_c', 
    'i_Automotive_c', 'i_Fitness_c', 'i_Health_c', 
    'i_Nightlife_c', 'i_Food_c', 'i_Shopping_c', 
    'i_Hotels_c', 'i_Restaurants_c'
]
# ===========================================

def analyze_overlap(file_path, cols):
    # 1. 读取数据
    try:
        df = pd.read_csv(file_path, sep='\t') 
    except:
        df = pd.read_csv(file_path, sep=',') # 备用尝试逗号

    print(f"数据加载成功，共 {len(df)} 个物品。")

    # 2. 数据转换：将 2 转为 1 (True)，1 转为 0 (False)
    # 逻辑：x == 2 为 True，转为 int 变成 1
    df_binary = df[cols].applymap(lambda x: 1 if x == 2 else 0)

    # 3. 统计每个类别的单独数量
    counts = df_binary.sum().sort_values(ascending=False)
    print("\n=== 各类别物品数量 (由多到少) ===")
    print(counts)

    # 4. 计算重叠矩阵 (Co-occurrence Matrix)
    # 矩阵点乘：(N_items, N_cats).T @ (N_items, N_cats) = (N_cats, N_cats)
    overlap_matrix = df_binary.T.dot(df_binary)

    # 5. 计算重叠率矩阵 (Overlap Ratio)
    # 定义：Overlap(A, B) / Min(Count(A), Count(B))
    # 意义：如果 A 和 B 重叠率 0.9，说明其中较小的那个集合几乎完全包含在较大的集合里
    ratio_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    
    for c1 in cols:
        for c2 in cols:
            intersection = overlap_matrix.loc[c1, c2]
            min_len = min(counts[c1], counts[c2])
            if min_len > 0:
                ratio_matrix.loc[c1, c2] = intersection / min_len
            else:
                ratio_matrix.loc[c1, c2] = 0.0

    print("\n=== 重叠矩阵 (对角线为自身数量，非对角线为共同拥有的物品数) ===")
    print(overlap_matrix)
    
    # 6. 自动筛选建议
    # 找出重叠率非常高的组合 (比如 > 0.5)，建议避开
    print("\n=== 高重叠风险警告 (重叠率 > 30%) ===")
    visited = set()
    for c1 in cols:
        for c2 in cols:
            if c1 == c2: continue
            if (c1, c2) in visited or (c2, c1) in visited: continue
            
            ratio = ratio_matrix.loc[c1, c2]
            if ratio > 0.3: # 阈值可调
                print(f"!! {c1} <-> {c2} : 重叠率 {ratio:.2%}")
                visited.add((c1, c2))

    return counts, ratio_matrix

# 运行分析
if __name__ == '__main__':
    # 模拟数据生成 (如果你没有文件，可以用这个测试)
    # data = {col: np.random.choice([1, 2], size=1000, p=[0.8, 0.2]) for col in category_cols}
    # df = pd.DataFrame(data)
    # df.to_csv('item_meta.txt', sep='\t', index=False)
    
    counts, ratios = analyze_overlap(FILE_PATH, category_cols)