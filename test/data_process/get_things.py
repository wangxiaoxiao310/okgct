import re


def findd(m, d):
    for k, v in d.items():
        if m in v:
            return k


def find_output_target(lines, num, l_n):
    # print(lines[num])
    for i in range(num + 1, len(lines)):
        if l_n in lines[i]:
            line = lines[i].replace(" ", "")
            l = line.split("=")[0]
            params = line.split("(")[1].split(")")[0].split(",")
            return l, params[0], params[1], i  # 返回output与target


def find_model(lines, output, res):
    tmp = output + "="
    for line in lines:
        ind = lines.index(line)
        line = line.strip().replace(" ", "")
        if tmp in line:
            m = line.split("=")[1].split("(")[0]
            data = line.split("(")[1].split(")")[0]
            return findd(m, res), data, ind


def main(file_path):
    mindspore_flag = 0
    with open(file_path, "r", encoding="UTF-8") as f:
        lines = f.readlines()  # 读取所有行
    # 删除所有注释防止干扰判断
    # 定义正则表达式，用于匹配单行注释和多行注释
    comment_pattern = re.compile(r'#.*?$|\'\'\'.*?\'\'\'|""".*?""""', re.DOTALL | re.MULTILINE)
    tmp = []
    for line in lines:
        code_without_comments = re.sub(comment_pattern, "", line)
        # print(code_without_comments)
        tmp.append(code_without_comments)
    lines = tmp

    with open("data_process/Loss.txt", "r", encoding="UTF-8") as ff:
        Loss_functions = ff.readlines()
    with open("data_process/mindspore_optim.txt", "r", encoding="UTF-8") as fff:
        mindspore_optims = fff.readlines()
    res = {}
    for i in range(0, len(Loss_functions)):
        Loss_functions[i] = Loss_functions[i].strip()
    for i in range(0, len(mindspore_optims)):
        mindspore_optims[i] = mindspore_optims[i].strip()
    # 处理导入的包，防止有别名
    packages = {}

    for line in lines:
        if "import" not in line:
            continue
        if "mindspore" in line:
            mindspore_flag = 1
        if "as" not in line and "from" not in line:
            continue

        line = line.strip()
        if "," in line:
            line = line.replace(",", " ")
        words = line.split()
        # print(words)
        head = ""
        if words[0] == "from":
            head = words[1] + "."
        words.append("**")
        for i in range(words.index("import") + 1, len(words) - 1):
            if words[i - 1] == "as" or words[i] == "as":
                continue
            if words[i + 1] != "as":
                packages[words[i]] = head + words[i]
                # print(words[i])
            else:
                packages[words[i + 2]] = head + words[i]
                # print(words[i+2])
    # 获取优化器和对应的模型
    optim_defs = []
    if mindspore_flag == 0:
        for line in lines:
            if "torch.optim" in line and "lr_scheduler" not in line:
                optim_defs.append(line.strip())
    else:
        for i in range(0, len(lines)):
            line = lines[i].replace(" ", "").strip()
            if "=" not in line:
                continue
            nam = line.split("=")[1].split("(")[0]
            tmp = nam.split(".")
            if tmp[0] in packages.keys():
                nam = nam.replace(tmp[0], packages[tmp[0]])
            if nam in mindspore_optims:
                optim_defs.append(lines[i].strip())
    # print(optim_defs)
    for optim_def in optim_defs:
        optim = optim_def.split("=")[0].strip()
        step_sentence = optim + ".step()"
        mod = optim_def.split("(")[1].split(".")[0]  # 定义优化器的第一个参数是模型的参数列表，因此可以知道该优化器对应哪个模型
        if "=" in mod:
            mod = mod.split("=")[1]
        for i in range(0, len(lines)):
            if step_sentence in lines[i]:
                res[i] = [""] * 8
                res[i][2] = optim
                res[i][0] = mod
    # print(res)
    # 搜寻损失函数，通过定义来寻找，损失函数的定义全在Loss.txt文件中，找到损失函数，接下来调用它的地方就能找到output和target，再回去找调用
    for i in range(0, len(lines)):
        line = lines[i].replace(" ", "").strip()
        if "=" not in line:
            continue
        nam = line.split("=")[1].split("(")[0]
        tmp = nam.split(".")
        if tmp[0] in packages.keys():
            nam = nam.replace(tmp[0], packages[tmp[0]])
        if nam in Loss_functions:
            loss_name = line.split("=")[0]
            loss_result, output, target, line_num2 = find_output_target(lines, i, loss_name)
            # print(output+'---'+target)
            index, data, line_num1 = find_model(lines, output, res)
            if index == None:
                continue
            res[index][1] = loss_name
            res[index][3] = data
            res[index][4] = target
            res[index][5] = loss_result
            res[index][6] = line_num1
            res[index][7] = line_num2
    # print(res)
    for k, v in res.items():  # 110: ['Model', 'criterion', 'optimizer', 'images', 'target', 'loss', 103, 104]
        v[1] = v[1] if v[1] != "" else "SoftIoULoss"
        v[3] = v[3] if v[3] != "" else "data"
        v[4] = v[4] if v[4] != "" else "labels"
        v[5] = v[5] if v[5] != "" else "loss"
        v[7] = v[7] if v[7] != "" else int(k) - 3
        line = lines[k]
        idx = line.find(v[2] + ".step")
        line = (
            line[:idx]
            + v[5]
            + " = API.train_step("
            + v[0]
            + ", "
            + v[1]
            + ", "
            + v[2]
            + ", "
            + v[3]
            + ", "
            + v[4]
            + ")\n "
        )
        lines[k] = line
        # lines[v[6]] = "#" + lines[v[6]]
        # print(int(v[7]))
        lines[int(v[7])] = "#" + lines[int(v[7])]
        # print(lines[k])
    return lines


if __name__ == "__main__":
    file_path = "../tmp/Image_Classification/train.py"
    lines = main(file_path)
    # for l in lines:
    #    print(l)
