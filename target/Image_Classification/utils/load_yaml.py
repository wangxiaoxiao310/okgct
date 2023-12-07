import sys
import yaml

sys.path.append("..")


def reload_args(file_path, parser):
    with open(file_path, mode="r", encoding="utf-8") as f:
        yamlConf = yaml.load(f.read(), Loader=yaml.FullLoader)
    parser.set_defaults(**yamlConf)
    p = parser.parse_args(args=[])
    return p
