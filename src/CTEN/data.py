'''
import torch

# 加载 .pt 文件
data = torch.load('/home/ubuntu/zzq/CTEN_jittor/output.pt', map_location='cpu')

# 打印结构信息
for key, value in data.items():
    print(f"{key}: {type(value)}, shape: {value.shape if isinstance(value, torch.Tensor) else 'N/A'}")

# 若你想看某个 tensor 的具体值（只显示前几个）
print("\nSample values:")
input=data['input_visual_reshaped']
sample=input[0,:,0,10:20,10:20]
print(sample)
#print(data['input_visual_reshaped'][0:1])  # 注意替换 'some_key'
'''
'''
print(data['resnet_output'][0:1]) 
print(data['a_resnet_output'][0:1]) 
print(data['nl_audio'][0:1]) 
'''     
'''
input_path = "/home/ubuntu/zzq/CTEN_jittor/outcome3.txt"
output_path = "/home/ubuntu/zzq/CTEN_jittor/outcome4.txt"

with open(input_path, "r") as infile, open(output_path, "w") as outfile:
    for line in infile:
        # 去除左右中括号和换行符
        line = line.strip().replace("[", "").replace("]", "")
        # 按逗号分隔每个数字
        numbers = line.split(",")
        # 写入每个数字到新的一行
        for num in numbers:
            outfile.write(num.strip() + "\n")
'''
file1_path = "/home/ubuntu/zzq/CTEN_jittor/outcome4.txt"
file2_path = "/home/ubuntu/zzq/CTEN_jittor/outcome2.txt"

with open(file1_path, "r") as f1, open(file2_path, "r") as f2:
    lines1 = [line.strip() for line in f1 if line.strip() != ""]
    lines2 = [line.strip() for line in f2 if line.strip() != ""]

max_len = max(len(lines1), len(lines2))
diff_count = 0

for i in range(max_len):
    val1 = lines1[i] if i < len(lines1) else "<MISSING>"
    val2 = lines2[i] if i < len(lines2) else "<MISSING>"
    if val1 != val2:
        diff_count += 1

print(f"Total number of differing lines: {diff_count}")
