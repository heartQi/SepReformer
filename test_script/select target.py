import torch.nn.functional as F
import torch
def read_data(data_file):
    # 用于存储读取的数据
    data_values = []
    # 逐行读取数据文件
    with open(data_file, 'r') as file:
        for line in file:
            # 去除空白字符，并去除开头和结尾的括号
            clean_line = line.strip().strip('tensor(').strip(')')

            # 将提取的字符串部分转换为浮点数
            try:
                value = float(clean_line)
                data_values.append(value)
            except ValueError as e:
                print(f"Error converting '{clean_line}' to float: {e}")
    return torch.tensor(data_values)

# 假设 .data 文件中的数据是空格分隔的
target_data = read_data('/Users/mervin.qi/Desktop/PSE/Workspace/voicefilter/devc/rc_win_rcadmin-uh34/dvec_rc_win_admin-uh34.data').unsqueeze(0)

paths = [
         "/Users/mervin.qi/Desktop/PSE/Workspace/voicefilter/devc/dvec-roman.data",
         "/Users/mervin.qi/Desktop/PSE/Workspace/voicefilter/devc/dvec-mia.data",
         "/Users/mervin.qi/Desktop/PSE/Workspace/voicefilter/devc/rc_win_rcadmin/dvec_rc_win_admin.data",
         "/Users/mervin.qi/Desktop/PSE/Workspace/voicefilter/devc/dvec-mervin.data",
         "/Users/mervin.qi/Desktop/PSE/Workspace/voicefilter/devc/Eria-mervin.data",
         "/Users/mervin.qi/Desktop/PSE/Workspace/voicefilter/devc/rc_win_rcadmin-uh34/dvec_rc_win_admin-uh34.data",
         "/Users/mervin.qi/Desktop/PSE/Workspace/voicefilter/devc/qmw-win/qmw-win.data",
         "/Users/mervin.qi/Desktop/PSE/Workspace/voicefilter/devc/rc_win_rcadmin-1/dvec_rc_win_admin-1.data",
         "/Users/mervin.qi/Desktop/PSE/Workspace/voicefilter/devc/rc_win_rcadmin-2/dvec_rc_win_admin-2.data"
         ]

# 初始化一个空数组来存储相似度值
similarities = []
# 遍历路径列表，读取每个文件并计算与目标数据的余弦相似度
for path in paths:
    data = read_data(path).unsqueeze(0)
    similarity = F.cosine_similarity(target_data, data).item()
    similarities.append(similarity)

print(similarities)