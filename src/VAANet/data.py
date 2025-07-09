
'''
input_path = "/home/ubuntu/wwc/zzq/VAANet_jt/outcome5.txt"
output_path = "/home/ubuntu/wwc/zzq/VAANet_jt/outcome6.txt"

with open(input_path, "r") as infile, open(output_path, "w") as outfile:
    for line in infile:
        # 去除左右中括号和换行符
        line = line.strip().replace("[", "").replace("]", "")
        # 按逗号分隔每个数字
        numbers = line.split(",")
        # 写入每个数字到新的一行
        for num in numbers:
            outfile.write(num.strip()+ "\n")
'''
file1_path = "/home/ubuntu/wwc/zzq/VAANet_jt/outcome6.txt"
file2_path = "/home/ubuntu/wwc/zzq/VAANet/outcome4.txt"

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
