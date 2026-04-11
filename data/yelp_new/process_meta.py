import pandas as pd
import numpy as np

def process_item_metadata(input_file, output_file):
    # 1. 读取原始 CSV
    # 注意：原始数据中可能有复杂的引号或逗号，pandas 通常能自动处理
    try:
        df = pd.read_csv(input_file)
        print(f"成功读取文件，共 {len(df)} 行数据")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 2. 处理 item_id (按照顺序排列，从 1 开始)
    # 重置索引，确保从 0 开始，然后 +1
    df = df.reset_index(drop=True)
    df['new_item_id'] = df.index + 1

    # 3. 定义目标分类列表 (根据你的要求)
    target_categories = [
        'Entertainment', 
        'Beauty', 
        'Home Services', 
        'Automotive', 
        'Fitness', 
        'Health', 
        'Nightlife', 
        'Food', 
        'Shopping', 
        'Hotels', 
        'Restaurants'
    ]

    # 4. 处理分类逻辑 (包含=2, 不包含=1)
    # 先处理空值，防止报错
    df['categories'] = df['categories'].fillna('')

    cat_columns = []
    for cat in target_categories:
        col_name = f'i_{cat}_c'
        cat_columns.append(col_name)
        
        # 核心逻辑：如果 categories 字符串包含该词，则为 2，否则为 1
        # case=False 表示忽略大小写
        df[col_name] = df['categories'].astype(str).apply(
            lambda x: 2 if cat.lower() in x.lower() else 1
        )

    # 6. 整理最终列顺序
    final_columns = ['item_id'] + cat_columns
    
    final_df = df[final_columns]
    
    # 重命名 new_item_id 为 item_id
    final_df = final_df.rename(columns={'new_item_id': 'item_id'})

    # 7. 保存文件
    # sep='\t' 表示用制表符分隔，这是学术界推荐系统数据集常用的格式
    final_df.to_csv(output_file, sep='\t', index=False)
    print(f"处理完成！文件已保存为: {output_file}")
    print("\n前5行预览：")
    print(final_df.head())

if __name__ == "__main__":
    # 输入文件名 (请确保你的文件名正确)
    input_csv = 'yelp_business.csv' 
    # 输出文件名
    output_csv = 'item_meta.csv'
    
    process_item_metadata(input_csv, output_csv)
