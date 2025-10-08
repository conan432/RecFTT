import pandas as pd
import os
import json
from collections import defaultdict
import ast 

def analyze_latent_effects_movielens():
    base_dir = '../../log/LightGCN_SAE/result_file/'
    item_info_path = '/thuir/jianguohao/ReChorus/data/ML_1MTOPK_new/item_info.csv'
    activation_path = os.path.join(base_dir, 'LightGCN_SAE__ML_1MTOPK_new__0__lr=0.001__l2=1e-08__emb_size=64__n_layers=3__batch_size=256__sae_lr=1e-05__sae_batch_size=16__sae_k=8__sae_scale_size=16test_activation.csv')
    prediction_template = 'LightGCN_SAE__ML_1MTOPK_new__0__lr=0.001__l2=1e-08__emb_size=64__n_layers=3__batch_size=256__sae_lr=1e-05__sae_batch_size=16__sae_k=8__sae_scale_size=16_prediction_ablation_{latent_id}-{scale}.csv'

    output_dir = './lightgcn_analyze/'
    os.makedirs(output_dir, exist_ok=True)

    print("--- 步骤 1: 开始加载和预处理数据 ---")

    try:
        item_info_df = pd.read_csv(item_info_path, sep='\t')
    except FileNotFoundError:
        print(f"错误: 物品信息文件未找到: {item_info_path}")
        return

    item_info_df['genres'] = item_info_df['genres'].fillna('')

    def parse_genres(genres_str):
        if not isinstance(genres_str, str) or not genres_str.strip():
            return []
        return [genre.strip() for genre in genres_str.split('|') if genre.strip()]

    item_to_categories = item_info_df.set_index('item_id')['genres'].apply(parse_genres).to_dict()
  
    category_counts = defaultdict(int)
    for categories in item_to_categories.values():
        for category in categories:
            category_counts[category] += 1
    
    all_categories = list(category_counts.keys())
    total_items = len(item_info_df)
    
    category_counts_path = os.path.join(output_dir, 'category_counts.json')
    with open(category_counts_path, 'w', encoding='utf-8') as f:
        sorted_category_counts = dict(sorted(category_counts.items(), key=lambda item: item[1], reverse=True))
        json.dump(sorted_category_counts, f, indent=4, ensure_ascii=False)
        
    print(f"物品总数: {total_items}")
    print(f"类别统计信息已保存至: {category_counts_path}")
    print("-" * 20)

    try:
        activation_df = pd.read_csv(activation_path, sep='\t')
    except FileNotFoundError:
        print(f"错误: 激活文件未找到: {activation_path}")
        return
        
    latent_to_users = defaultdict(list)
    for _, row in activation_df.iterrows():
        user_id = row['user_id']
        try:
            indices = ast.literal_eval(row['indices'])
            for latent_id in indices:
                latent_to_users[latent_id].append(user_id)
        except (ValueError, SyntaxError):
            print(f"警告: 无法解析 user_id {user_id} 的 indices: {row['indices']}")
            continue

    print("激活数据加载完成，已建立 latent 到 users 的映射。")
    print("--- 步骤 1 完成 ---")
    print("\n--- 步骤 2: 开始遍历所有潜在因子并计算得分 ---")

    results_by_category = defaultdict(list)
    
    for latent_id in range(1024):
        if latent_id % 50 == 0:
            print(f"正在处理 Latent ID: {latent_id} / 1023")

        activated_users = latent_to_users.get(latent_id)
        if not activated_users:
            continue
        
        case_count = len(activated_users)
        
        file_10_0_path = os.path.join(base_dir, prediction_template.format(latent_id=latent_id, scale=10.0))
        file_0_1_path = os.path.join(base_dir, prediction_template.format(latent_id=latent_id, scale=0.1))

        try:
            df_10_0 = pd.read_csv(file_10_0_path, sep='\t')
            df_0_1 = pd.read_csv(file_0_1_path, sep='\t')
        except FileNotFoundError:
            continue

        df_10_0_filtered = df_10_0[df_10_0['user_id'].isin(activated_users)]
        df_0_1_filtered = df_0_1[df_0_1['user_id'].isin(activated_users)]

        def calculate_total_proportions(df):
            total_proportions = defaultdict(float)
            for _, row in df.iterrows():
                try:
                    rec_items = ast.literal_eval(row['rec_items'])
                    rec_len = len(rec_items)
                    if rec_len == 0:
                        continue
                    
                    category_counts_in_rec = defaultdict(int)
                    for item in rec_items:
                        categories = item_to_categories.get(item, [])
                        for category in categories:
                            category_counts_in_rec[category] += 1
                    
                    for category, count in category_counts_in_rec.items():
                        total_proportions[category] += count / rec_len
                except (ValueError, SyntaxError):
                    print(f"警告: 无法解析 user_id {row['user_id']} 的 rec_items: {row['rec_items']}")
                    continue
            return total_proportions

        total_proportions_10_0 = calculate_total_proportions(df_10_0_filtered)
        total_proportions_0_1 = calculate_total_proportions(df_0_1_filtered)

        for category in all_categories:
            avg_ratio_x10 = total_proportions_10_0.get(category, 0.0) / case_count
            avg_ratio_x0_1 = total_proportions_0_1.get(category, 0.0) / case_count
            score = avg_ratio_x10 - avg_ratio_x0_1
            
            if score != 0 or avg_ratio_x10 != 0 or avg_ratio_x0_1 != 0:
                results_by_category[category].append({
                    "latent_id": latent_id,
                    "score": score,
                    "avg_ratio_x10": avg_ratio_x10,
                    "avg_ratio_x0_1": avg_ratio_x0_1,
                    "case_count": case_count
                })

    print("--- 步骤 2 完成 ---")
    print("\n--- 步骤 3: 开始生成并保存 JSON 结果文件 ---")

    for category, results_list in results_by_category.items():
        safe_category_name = "".join(c if c.isalnum() else '_' for c in category)
        output_filename = os.path.join(output_dir, f"{safe_category_name}_regulation_scores.json")
        
        results_list.sort(key=lambda x: x['score'], reverse=True)
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=4, ensure_ascii=False)
        
        print(f"已生成文件: {output_filename}")

    print("--- 步骤 3 完成 ---")
    print(f"\n所有任务已完成！结果文件已保存在 '{output_dir}' 文件夹中。")


if __name__ == '__main__':
    analyze_latent_effects_movielens()