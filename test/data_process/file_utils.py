import os
import re
import shutil


class Stack:
    def __init__(self):
        self.stack = []

    def push(self, val):
        self.stack.append(val)

    def pop(self):
        return self.stack.pop()

    def top(self):
        return self.stack[-1]

    def size(self):
        return len(self.stack)

    def isEmpty(self):
        return not self.stack


def is_alpha(string):
    key = ["True", "False", "None"]
    if string in key:
        return False
    pattern = re.compile(r"[a-zA-Z]")
    return bool(pattern.match(string))


def get_arglist_from_str(arg_string):
    # print('arg_string:', arg_string)
    """
    根据参数str(不包含外括号)，得到每个参数及其对应值

    :param params_string:
    :return: [[param1,value1], [param2,value2]]
    """
    # print(arg_string)
    arglist = []
    stack = Stack()
    s = ""
    for i in range(len(arg_string) + 1):
        if (i == len(arg_string) or arg_string[i] == ",") and stack.isEmpty():
            s = s.strip()
            if s.find("=") != -1 and s.find("=") < s.find("("):
                # 只有()没有[] 或者 既有[]也有()
                arg = s.split("=", maxsplit=1)
                arglist.append([arg[0], arg[1]])
            elif s.find("=") != -1 and s.find("=") < s.find("[") and s.find("(") == -1:
                # 只有[]没有()
                arg = s.split("=", maxsplit=1)
                arglist.append([arg[0], arg[1]])
            elif s.find("=") != -1 and s.find("(") == -1 and s.find("[") == -1:
                # 既没有()又没有[]
                arg = s.split("=", maxsplit=1)
                arglist.append([arg[0], arg[1]])
            else:
                arglist.append(["None", s])
            s = ""
        elif arg_string[i] == "[" or arg_string[i] == "(":
            stack.push(arg_string[i])
            s = s + arg_string[i]
        elif arg_string[i] == "]" or arg_string[i] == ")" and not stack.isEmpty():
            stack.pop()
            s = s + arg_string[i]
        else:
            s = s + arg_string[i]

    # print('arglist:', arglist)
    return arglist


def project_copy(primary_dir, target_dir):  # 拷贝方法 把原始文件夹的所有文件夹和文件 按照同样的名字拷贝到目标文件夹中
    """
    将文件夹 primary_dir 整个复制到 target_dir
    Parameters
    ----------
    primary_dir
    target_dir
    """
    # 遍历filepath下所有文件,包括目录
    files = os.listdir(primary_dir)
    for i in files:  # i 是目录下的文件或文件夹
        if (
            (not (i.find(".git") == -1)) or (not (i.find(".pyc") == -1)) or (not (i.find(".out") == -1))
        ):  # 不复制.git,.pyc, .out
            continue
        i = os.path.join(primary_dir, i)  # 字符串拼接
        i_new = os.path.join(target_dir, os.path.basename(i))  # 目标文件夹也要改变
        if os.path.isdir(i):  # 如果是文件夹
            if not os.path.exists(i_new):  # 如果没新建过 新建同名目标文件夹
                os.makedirs(i_new)
            project_copy(i, i_new)  # 递归循环下一个目录 复制目录里面的内容
        else:  # 不是文件夹 文件 判断字符串是否有_bin 粘贴到指定位置 并且修改名字
            oldname = i
            newname = i_new
            if os.path.basename(i) == "train.py":
                shutil.copyfile(
                    os.path.join(os.path.curdir, "data_process/" + "API.py"), os.path.join(target_dir, "API.py")
                )
            if os.path.basename(i) == "VIT.py":
                shutil.copyfile(
                    os.path.join(os.path.curdir, "data_process/" + os.path.basename(i)),
                    os.path.join(target_dir, os.path.basename(i)),
                )
            if not os.path.exists(newname):  # 如果文件不存在,存在了就不拷贝了
                print(oldname, newname)
                shutil.copyfile(oldname, newname)
