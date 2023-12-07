from data_process.process import solve_project

SOURCE_PATH = "source_code/"
TARGET_PATH = "target_code/"
NET = "resnet50.py"


def traversal_project(folder_name):
    """
    dirpath是文件夹路径
    dirnames是文件夹名称
    filenames是文件名称
    :param folder_name:
    :return:
    """

    framework = "torch"
    project_path = "../origin"
    solve_project(project_path, framework)


if __name__ == "__main__":
    traversal_project("source_code")
