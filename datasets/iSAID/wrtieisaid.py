with open('/home/buaad208/zhangyfcode/256pots+DINO/datasets/iSAID/val.txt', 'r') as f:
    lines = f.readlines()

cleaned_lines = [line.replace('_instance_color_RGB', '').strip() for line in lines]


for line in cleaned_lines:
    print(line)


with open('/home/buaad208/zhangyfcode/256pots+DINO/datasets/iSAID/val1.txt', 'w') as f:
    f.write('\n'.join(cleaned_lines))