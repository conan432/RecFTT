import pandas as pd
import os
import json
from collections import defaultdict
import ast 

def analyze_latent_effects_grocery():
    data_base_dir = '/thuir/jianguohao/ReChorus/data/Grocery_and_Gourmet_Food/'
    result_base_dir = '../../log/LightGCN_SAE/result_file/'
    item_info_dir = os.path.join(data_base_dir, "grouped_items")
    activation_path = os.path.join(result_base_dir, 'LightGCN_SAE__Grocery_and_Gourmet_Food__0__lr=0.001__l2=1e-08__emb_size=64__n_layers=3__batch_size=256__sae_lr=5e-05__sae_batch_size=4__sae_k=8__sae_scale_size=16test_activation.csv')

    prediction_template = 'LightGCN_SAE__Grocery_and_Gourmet_Food__0__lr=0.001__l2=1e-08__emb_size=64__n_layers=3__batch_size=256__sae_lr=5e-05__sae_batch_size=4__sae_k=8__sae_scale_size=16_prediction_ablation_{latent_id}-{scale}.csv'

    output_dir = './lightgcn_analyze/'
    os.makedirs(output_dir, exist_ok=True)

    separator = '\t'

    print("--- 步骤 1: 开始加载和预处理数据 ---")

    item_to_categories = defaultdict(list)
    category_to_items = defaultdict(set)

    feature_files = [
        ("gluten_free_items.csv", "is_gluten_free", 1, "gluten_free"),
        ("price_group_items.csv", "price_group", 'Low price', "low_price"),
        ("caffeinated_items.csv", "is_caffeinated", 0, "caffeine_free"),
        ("vegetarian_items.csv", "is_vegetarian", 1, "vegetarian"),
        ("no_sugar_items.csv", "is_no_sugar", 1, "no_sugar"),
        ("age_group_items.csv", "age_group", 'Children-Friendly', "children_friendly")
    ]
    
    print("正在加载 Grocery 数据集的物品元数据...")
    for file_name, col_name, target_value, category_name in feature_files:
        path = os.path.join(item_info_dir, file_name)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, sep=separator)
                target_items = set(df[df[col_name] == target_value]['item_id'])
                category_to_items[category_name] = target_items

                for item_id in target_items:
                    item_to_categories[item_id].append(category_name)
                print(f"  - 已加载特征 '{category_name}'，包含 {len(target_items)} 个物品。")
            except Exception as e:
                print(f"警告: 加载文件 {file_name} 时出错: {e}")
        else:
            print(f"警告: 元数据文件未找到: {path}")

    category_counts = {category: len(items) for category, items in category_to_items.items()}
    all_categories = list(category_counts.keys())
    total_items = len(set.union(*category_to_items.values())) if category_to_items else 0
    
    category_counts_path = os.path.join(output_dir, 'category_counts.json')
    with open(category_counts_path, 'w', encoding='utf-8') as f:
        sorted_category_counts = dict(sorted(category_counts.items(), key=lambda item: item[1], reverse=True))
        json.dump(sorted_category_counts, f, indent=4, ensure_ascii=False)
        
    print(f"物品总数 (在已加载特征中): {total_items}")
    print(f"类别统计信息已保存至: {category_counts_path}")
    print("-" * 20)

    try:
        activation_df = pd.read_csv(activation_path, sep=separator)
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
        
        file_10_0_path = os.path.join(result_base_dir, prediction_template.format(latent_id=latent_id, scale=10.0))
        file_0_1_path = os.path.join(result_base_dir, prediction_template.format(latent_id=latent_id, scale=0.1))

        try:
            df_10_0 = pd.read_csv(file_10_0_path, sep=separator)
            df_0_1 = pd.read_csv(file_0_1_path, sep=separator)
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
    analyze_latent_effects_grocery()