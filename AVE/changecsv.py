import csv

# 输入文件名（包含按照&分隔的数据）
input_filename = r'D:\Multimodal\Dataset\AVE_Dataset\Annotations.txt'
# 输出文件名（CSV格式）
output_filename = 'Annotations.csv'

# 定义CSV文件的列名
fieldnames = ['Category', 'VideoID', 'Quality', 'StartTime', 'EndTime']

# 打开输入文件进行读取
with open(input_filename, 'r', encoding='utf-8') as infile:
    # 打开输出文件进行写入，并指定fieldnames作为表头
    with open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        # 写入表头
        writer.writeheader()

        # 逐行读取输入文件
        for line in infile:
            # 按照&分隔符拆分每行数据
            parts = line.strip().split('&')

            # 创建一个字典，将拆分后的数据映射到CSV的列名上
            row_dict = {
                'Category': parts[0],
                'VideoID': parts[1],
                'Quality': parts[2],
                'StartTime': parts[3],
                'EndTime': parts[4]
            }

            # 将字典写入CSV文件
            writer.writerow(row_dict)

print(f"数据已成功转换为CSV文件，并保存为{output_filename}")