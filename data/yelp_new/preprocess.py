import pandas as pd

# 1. 读取 CSV 文件
# 假设你的文件名为 'input.csv'
# 如果你的文件是用逗号分隔的，直接用 read_csv
# 如果你的文件是用 Tab (制表符) 分隔的（看起来你的示例像 Tab 分隔），请加上 sep='\t' 参数
input_file = 'train.csv'
output_file = 'sorted_train.csv'

try:
    # 读取数据
    # 这里的 sep='\t' 是为了应对你的示例文本看起来像 Tab 分隔
    # 如果你的文件确实是逗号分隔的标准 CSV，请去掉 sep='\t'，变成 pd.read_csv(input_file)
    df = pd.read_csv(input_file, sep='\t') 
    
    # 为了保险起见，如果读取第一行发现只有1列（说明分隔符不对），尝试用逗号读取
    if df.shape[1] == 1:
        df = pd.read_csv(input_file, sep=',')

    print("原始数据预览：")
    print(df.head())

    # 2. 排序
    # by=['user_id', 'time']: 先按 user_id 排，如果相同则按 time 排
    # ascending=[True, True]: 两个字段都按升序（从小到大）排列
    df_sorted = df.sort_values(by=['user_id', 'time'], ascending=[True, True])

    print("\n排序后数据预览：")
    print(df_sorted.head())

    # 3. 保存为新的 CSV 文件
    # index=False 表示不把 pandas 的行索引（0,1,2...）写入文件
    # sep='\t' 表示保持用 Tab 分隔保存，如果你想转成标准逗号 CSV，去掉 sep 参数即可
    df_sorted.to_csv(output_file, index=False, sep='\t')

    print(f"\n成功！排序后的文件已保存为: {output_file}")

except FileNotFoundError:
    print(f"错误：找不到文件 {input_file}，请确保文件名正确。")
except Exception as e:
    print(f"发生错误：{e}")