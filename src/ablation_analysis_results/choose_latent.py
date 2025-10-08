import json
import argparse
import os

def generate_ablation_config(score_file_path, output_file_path, top_x, boost_scale=1.5):
    try:
        with open(score_file_path, 'r') as f:
            data = json.load(f)
        print(f"成功读取 {len(data)} 条记录于 '{score_file_path}'。")
    except FileNotFoundError:
        print(f"错误：文件未找到于 '{score_file_path}'。")
        return
    except json.JSONDecodeError:
        print(f"错误：无法解析 '{score_file_path}' 中的 JSON 数据。")
        return

    # 1. 划分正分、负分和零分的 latent
    positive_latents = []
    negative_latents = []

    for item in data:
        score = item.get('score', 0)
        if score > 0:
            positive_latents.append(item)
        elif score < 0:
            negative_latents.append(item)
    
    print(f"划分结果：正分 latent 数量 = {len(positive_latents)}, 负分 latent 数量 = {len(negative_latents)}")

    # 2. 对分组进行排序
    # 正分组按 score 降序（从最好到最差）
    positive_latents.sort(key=lambda x: x['score'], reverse=True)
    # 负分组按 score 升序（从最差到最好，即绝对值最大到最小）
    negative_latents.sort(key=lambda x: x['score'], reverse=False)

    # 3. 准备最终的 ablation 列表
    ablations = []
    weaken_scale = 1.0 / boost_scale

    # 4. 选取正分前 X 个进行增强
    selected_positive = positive_latents[:top_x]
    print(f"\n选择 Top {len(selected_positive)} 个正分 latent 进行增强 (scale = {boost_scale}):")
    for item in selected_positive:
        ablations.append(item['latent_id'])
        print(f"  - Latent ID: {item['latent_id']}, Score: {item['score']:.4f}")

    # 5. 选取负分前 X 个（即分数最低的X个）进行削弱
    selected_negative = negative_latents[:top_x]
    print(f"\n选择 Top {len(selected_negative)} 个负分 latent 进行削弱 (scale = {weaken_scale:.4f}):")
    for item in selected_negative:
        ablations.append(item['latent_id'])
        print(f"  - Latent ID: {item['latent_id']}, Score: {item['score']:.4f}")

    # 6. 构造最终的 JSON 对象
    output_data = {
        "comment": f"Ablation config generated from {os.path.basename(score_file_path)}. "
                   f"Boosting top {top_x} positive and weakening top {top_x} negative latents.",
        "ablations": ablations
    }

    # 7. 写入文件
    try:
        with open(output_file_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n成功生成调控文件，已保存到: '{output_file_path}'")
    except Exception as e:
        print(f"\n错误：写入文件 '{output_file_path}' 时失败。错误信息: {e}")

if __name__ == '__main__':
    # 使用 argparse 使脚本可以通过命令行参数运行，更加灵活
    parser = argparse.ArgumentParser(description="Generate SAE ablation config from a score file.")
    parser.add_argument("score_file", type=str, help="Path to the input JSON score file.")
    parser.add_argument("output_file", type=str, help="Path for the output ablation JSON file.")
    parser.add_argument("-x", "--top_x", type=int, default=5, help="Number of top latents to select from each group (positive and negative). Default is 5.")
    parser.add_argument("-s", "--scale", type=float, default=1.5, help="Boost scale for positive latents. Weaken scale will be 1/scale. Default is 1.5.")
    
    args = parser.parse_args()
    
    generate_ablation_config(
        score_file_path=args.score_file,
        output_file_path=args.output_file,
        top_x=args.top_x,
        boost_scale=args.scale
    )