# 读取原始文件并处理
with open('/home/buaad208/zhangyfcode/256pots+DINO/datasets/iSAID/train.txt', 'r') as infile, open('/home/buaad208/zhangyfcode/256pots+DINO/datasets/iSAID/train11.txt', 'w') as outfile:
    for line in infile:
        # 分割每行，取第一部分（图像名称）
        image_name = line.split('\t')[0]
        # 写入到输出文件
        outfile.write(image_name + '\n')