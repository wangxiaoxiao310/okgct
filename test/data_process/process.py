"""
对pytorch文件预处理，补全代码并获取torch算子
todo:
1. tansfer 时，对于 “” 字符串里面的应该不该，这里没处理

"""
from data_process.file_utils import is_alpha, get_arglist_from_str
import os
import re

from data_process import file_utils
from okg.knowledgeGraph import knowledgeGraph
from data_process.post_process import solve

import_as_dict = {}
stop_word = [" ", "(", ")", "[", "]", "\t", "\n", ":", "=", ",", "."]  # 保证'.'在最后一个元素！
curent_fun = []
new_fun = []
kgraph = knowledgeGraph(clear=False)


def modify_api(func_name, full_arglist):
    # 删除 torch.cuda
    if func_name.startswith("torch.cuda."):
        return ""
    if func_name.endswith("_"):
        func_name = func_name[:-1]
    if full_arglist is None:
        if func_name.find("torch") != -1:
            func_name = func_name.replace("torch", "mindspore")
            func_name = func_name.replace("Module", "Cell")
            func_name = func_name.replace("Sequential", "SequentialCell")
            return func_name
        ms_op = kgraph.getMSInfoFromTorch(func_name)[0]
        if ms_op is not None:
            ms_name = ms_op.get("full_name")
            return ms_name
        return "****"

    # print(func_name, "  :  ", full_arglist)

    # 1. 补全参数信息
    top, tordered_ms_params, treturn, _ = kgraph.getOperatorInfo("pytorch", func_name, "1.5.0")
    if top is None:
        top, tordered_ms_params, treturn, _ = kgraph.getOperatorInfo("pytorch", func_name, "1.8.1")
        # if top is None:
        # print("pytorch 中无 %s 对应算子" % (func_name))

    if full_arglist is not None and tordered_ms_params is not None and len(full_arglist) <= len(tordered_ms_params):
        if tordered_ms_params[0].name != "args":
            for i in range(len(full_arglist)):
                if full_arglist[i][0] == "None":
                    full_arglist[i][0] = tordered_ms_params[i].name
    # 2. 获取对应mindspore算子信息
    msop, ordered_ms_params, msTorchRelation = kgraph.getMSInfoFromTorch(func_name)
    if msop is None:
        return "****"
        # print()

    ms_code = msop.get("full_name") + "("
    type_judge = False
    if msTorchRelation["type_judgement"] == "true":
        ms_code += ")("
        type_judge = True
    if ordered_ms_params is not None and len(ordered_ms_params) > 0 and ordered_ms_params[0].name == "args":
        for params in full_arglist:
            ms_code = ms_code + params[1] + ", "
    else:
        for ms_param in ordered_ms_params:
            ms_value = (
                '"' + ms_param.default + '"'
                if (is_alpha(ms_param.default) and ms_param.default.find("mindspore") == -1)
                else ms_param.default
            )
            if ms_param.name in msTorchRelation:
                torch_name = msTorchRelation[ms_param.name]
                for t_param in full_arglist:
                    if t_param[0] == torch_name:
                        ms_value = t_param[1]
                        break
            if type_judge:
                ms_code = ms_code + ms_value + ", "
            else:
                ms_code = ms_code + ms_param.name + "=" + ms_value + ", "
    if ms_code[-2] == ",":
        ms_code = ms_code[:-2]
    ms_code = ms_code + ")"
    # print(ms_code)
    return ms_code


def transfer_line(line, framework):
    # print(line)
    """
    string 级代码修改(line 是带有)：
        1. 每行只一个函数调用
    :type line: 每一行code
    """
    # 处理注释和函数定义行： 直接返回不处理
    start_line = line.strip()
    if start_line.startswith("#") or line.find(framework) == -1:
        return line

    new_line = line
    framework_index = new_line.find(framework)
    while framework_index != -1:
        i = framework_index
        while i < len(new_line) and (new_line[i].isalnum() or new_line[i] == "." or new_line[i] == "_"):
            i += 1
        api_name = new_line[framework_index:i]
        while i < len(new_line) and new_line[i] == " ":
            i += 1

        if i == len(new_line) or new_line[i] != "(":  # 非函数，则只改名
            flag = 1
            api_ms_name = modify_api(api_name, None)
            if api_ms_name == "****":
                return "# ****" + api_ms_name + "\n"
            new_line = new_line.replace(api_name, api_ms_name, 1)
            if flag:
                break
        else:
            bracket_left = i
            bracket_right = bracket_left
            sum = 0
            while bracket_right < len(new_line):
                if new_line[bracket_right] == ")":
                    sum = sum - 1
                if new_line[bracket_right] == "(":
                    sum = sum + 1
                if sum == 0:
                    break
                bracket_right = bracket_right + 1
            # 得到每个参数
            arg_str = new_line[bracket_left + 1 : bracket_right]
            arglist = get_arglist_from_str(arg_str)
            ms_func_code = modify_api(api_name, arglist)
            if ms_func_code == "****":
                return "# ****" + api_name + "\n"
            new_line = new_line.replace(new_line[framework_index : bracket_right + 1], ms_func_code, 1)

            if api_name not in curent_fun and api_name not in new_fun:  # 用于获取 API 的
                new_fun.append(api_name)
        framework_index = new_line.find(framework)

    return new_line


def get_arglist_str(arglist):
    result = "(" if len(arglist) > 0 else ""
    for i in range(len(arglist)):
        if i != 0:
            result = result + ", "
        result = result + arglist[i][0] + "=" + arglist[i][1]
    result = result + ")" if len(arglist) > 0 else ""
    return result


def process_import(line, framework):
    """
    处理源代码 import
    :param line: 每一行代码
    :return: 修改后的每一行代码
    """
    # 处理 import, 得到 import dict
    """
    import A
    import A as a
    import A , B as b
    import A as a, B as b
    from A import B
    from A import B as b
    from A import B, C as c
    1. 判断有无 from A, 有则记录前缀 A 
    2. import 后面语句按 , 分割，没有 as 的则 import_dic[B]=(A.)B,若无A则pass,有 as ,则 import_dic[b]=(A.)B
    """
    matchObj = re.match(r".*import (.*)", line)
    if matchObj:  # 提取 import 关系
        # print(line)
        if line.find(framework) != -1:
            prefix = ""  # A
            words = line.strip().split()
            word_index_from = words.index("from") if ("from" in words) else -1
            if word_index_from != -1:  # 有 from：from * import * (as *)
                prefix = words[word_index_from + 1]
            import_after_line = line[line.find("import") + 7 :].strip()  # B, C as c
            words = import_after_line.strip().split(",")
            import_package = []
            for word in words:
                word = word.strip()
                w = word.split()
                import_package.append(w[0].split(".")[0])
                w_index_as = w.index("as") if ("as" in w) else -1
                if w_index_as != -1:  # C as c
                    import_as_dict[w[2]] = w[0] if prefix == "" else prefix + "." + w[0]
                else:  # c
                    if not prefix == "":
                        import_as_dict[w[0]] = prefix + "." + w[0]
            if line.find("from") != -1:
                line = line[: line.find("from")] + "import " + prefix.split(".")[0] + "\n"
            else:
                # 组装 import
                line = line[: line.find("import")] + "import "
                for package in import_package:
                    line += package + ", "
                if line.endswith(", "):
                    line = line[:-2] + "\n"

    else:  # 修改非 import 行
        for alias in import_as_dict:
            alias_index = line.find(alias)
            while alias_index != -1:  # alias应为函数调用前缀
                if (alias_index + len(alias) == len(line) or line[alias_index + len(alias)] in stop_word) and (
                    alias_index == 0 or line[alias_index - 1] in stop_word[:-1]
                ):
                    subLine = line[alias_index:]
                    subLine = subLine.replace(alias, import_as_dict[alias], 1)
                    line = line[:alias_index] + subLine
                    alias_index = line.find(alias, alias_index + len(import_as_dict[alias]))
                else:  # 不修改
                    alias_index = line.find(alias, alias_index + len(alias))
    return line


def data_process(file_name, framework):
    """
    还原算子全称，写入临时文件
    删除#注释 与 """ """注释
    Parameters
    ----------
    file_name
    framework
    """
    # 每个文件有自己的 import_as_dict
    global import_as_dict
    import_as_dict = {}
    new_code = []

    # print('源代码：')
    with open(file_name, encoding="utf-8") as file:
        lines = file.readlines()
        # 删除注释
        i = 0
        new_lines = []
        while i < len(lines):
            # 删除三个单引号注释
            if lines[i].strip().startswith("'''"):
                if lines[i].strip().endswith("'''") and lines[i].strip() != "'''":  # """和"""在同一行
                    i += 1
                else:
                    i += 1
                    while i < len(lines):
                        if lines[i].strip().endswith("'''"):
                            i += 1
                            break
                        i += 1
            if i >= len(lines):
                break
            # 删除三个双引号注释
            if lines[i].strip().startswith('"""'):
                if lines[i].strip().endswith('"""') and lines[i].strip() != '"""':  # """和"""在同一行
                    i += 1
                else:
                    i += 1
                    while i < len(lines):
                        if lines[i].strip().endswith('"""'):
                            i += 1
                            break
                        i += 1
            if i >= len(lines):
                break
            # 删除#注释
            anoIndex = lines[i].find("#")
            if anoIndex != -1:
                lines[i] = lines[i][:anoIndex] + "\n"
                if not lines[i].strip():
                    lines[i] = ""
            # 删除 -> 注释
            # arrowIndex = lines[i].find("->")
            # if arrowIndex != -1:
            #     lines[i] = lines[i][:arrowIndex] + ':\n'
            new_lines.append(lines[i])
            i += 1

        i = 0
        while i < len(new_lines):
            line = process_import(new_lines[i], framework)  # 还原算子全称
            # 处理函数参数分行情况的补全，最后还是分行显示
            if line.count("(") != line.count(")"):
                sum_bracket = line.count("(") - line.count(")")
                while i < len(new_lines) and sum_bracket != 0:
                    i += 1
                    line_cur = process_import(new_lines[i], framework).lstrip()
                    line = line.rstrip() + line_cur
                    sum_bracket += line_cur.count("(") - line_cur.count(")")
            new_code.append(line)
            i += 1

    # 将代码写入目标文件
    with open(file_name, "w", encoding="utf-8") as f:
        for line in new_code:
            # print(line, end='')
            f.write(line)


def data_trasfer(file_name, framework):
    """
    函数修改
    Parameters
    ----------
    file_name
    framework
    """
    new_line = []
    with open(file_name, encoding="utf-8") as file:
        # 预处理：删除某些算子
        torch_packages = ["torchvision", "torchtext", "torchsummary", "torch"]
        for line in file.readlines():
            if line.strip().replace("import ", "") in torch_packages:
                for package in torch_packages:
                    line = line.replace(package, "mindspore")
                if line.strip() == "import torch":
                    line = line.replace("torch", "mindspore")
            else:
                line = transfer_line(line, framework)
            new_line.append(line)
    with open(file_name, "w", encoding="utf-8") as f:
        for line in new_line:
            f.write(line)


def process_and_transfer_project(path, framework):
    """
    对 project 下的文件做 torch -> 预处理 -> 补全
    Parameters
    ----------
    path
    framework
    """
    if os.path.exists(path):
        file_list = os.listdir(path)
        for f in file_list:
            f = os.path.join(path, f)
            if os.path.isdir(f):
                process_and_transfer_project(f, framework)
            else:
                file_name, extension = os.path.splitext(f)
                if extension == ".py":
                    print("\n", f)
                    data_process(f, framework)
                    data_trasfer(f, framework)


def solve_project(project_path, framework, is_get_api=False):
    """
    处理 Project 下所以 py 文件，预处理 + 转换
        1. 复制整个项目到 tmp/target 下，存在的不拷贝
        2. 预处理，读写到源文件
        3. 转换，读写到源文件
    is_get_api = True 则还会获取 project 下的所有 torch API，写入文件
    """

    dir_path = "tmp/" + os.path.basename(project_path)
    file_utils.project_copy(project_path, dir_path)
    process_and_transfer_project(dir_path, framework)
    solve(dir_path)
