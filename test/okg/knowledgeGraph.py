import re
import json
import string
import sys
from operator import attrgetter
from pathlib import Path

from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher


class Params:
    def __init__(self, name, order, type="", default=""):
        self.name = name
        self.order = order
        self.type = type
        self.default = default


class knowledgeGraph:
    def __init__(self, clear=False):
        self.nodes = {}  # 节点: label
        self.g = self.connect_neo4j()
        self.nodes_matcher = NodeMatcher(self.g)
        self.relationship_matcher = RelationshipMatcher(self.g)
        self.delete_query = "MATCH (n) DETACH DELETE n"  # 清空图谱指令 如果clear为True 则清空图谱
        self.delete_alone_nodes = "MATCH (n) WHERE size((n)--())=0 DELETE(n)"  # 清除孤立点
        if clear:
            self.clear_before_build()

    @staticmethod
    def connect_neo4j(username="neo4j", password="123456"):
        my_graph = Graph("bolt://neo4j@localhost:7687", auth=(username, password))
        return my_graph

    def clear_before_build(self):
        try:
            self.g.run(self.delete_query).evaluate()
            print("cleared all")
        except Exception as e:
            print(str(e))

    def delete_alone(self):
        try:
            self.g.run(self.delete_alone_nodes).evaluate()
        except Exception as e:
            print(str(e))

    def transmit_label(self):  # 给api节点也打上framework标签
        framework = ["PyTorch", "Paddle", "Mindspore", "TensorFlow"]
        for frm in framework:
            transmit_label_cypher = (
                f"MATCH p=(n:api)<-[*2]-(m:Framework) where m.name='{frm}' set n:`{frm}` "
                f"RETURN n.name as name, labels(m) as labels, m.name as framework"
            )
            self.run_cypher([transmit_label_cypher])

    def set_ka_label(self):  # 修正节点 label 和keyword argument关系相连的应该是keyword argument节点
        cypher = "MATCH p=()-[r:`keyword argument`]->(m:parameter) remove m:parameter set m:`keyword argument`"
        self.run_cypher([cypher])

    # 更正：label改为*label
    def node_creator(self, key, *label, **properties):
        # 根据key 查找相应的create_node_ 方法, 没找到就调用create_node方法
        method = "create_node_" + key.replace(" ", "")
        creator = getattr(self, method, self.create_node)
        return creator(*label, **properties)

    def create_node(self, label, **properties):
        """创建结点，如果结点有类型和属性的话，也一起创建"""
        node = Node(label, **properties)
        return node

    def create_node_api(self, class_label, is_class, **properties):
        """创建api结点，label：'api', 所属类别, 是否为class(True/False) 属性：api名"""
        if is_class is True:
            label = "class"
        else:
            label = "method"
        if class_label == "":
            node = Node("api", label, **properties)
        else:
            node = Node("api", class_label, label, **properties)
        return node

    def create_node_parameter(self, is_class, **properties):
        """创建parameter结点，label：'parameter', 'class'/'method' 属性：parameter名"""
        if is_class is True:
            node = Node("parameter", "class", **properties)
        else:
            node = Node("parameter", "method", **properties)
        return node

    def create_node_keywordargument(self, is_class, **properties):
        """创建keyword argument结点，label：'keyword argument', 'class'/'method' 属性：keyword argument名"""
        if is_class is True:
            node = Node("keyword argument", "class", **properties)
        else:
            node = Node("keyword argument", "method", **properties)
        return node

    def create_node_framework(self, version_label, **properties):
        """创建Framework结点，label：'Framework', 版本, 属性：Framework名"""
        node = Node("Framework", version_label, **properties)
        return node

    def create_node_package(self, framework_label, **properties):
        """创建Package结点，label：'Package', 所属Framework, 属性：Package名"""
        node = Node("Package", framework_label, **properties)
        return node

    def create_node_dtype(self, type_label, **properties):
        node = Node("dtype", type_label, **properties)
        return node

    def create_relationship(self, start_node: Node, relation_type: str, end_node: Node, relation_properties=None):
        """创建关系，如果有关系上属性的话就一起创建"""
        new_relation = Relationship(start_node, relation_type, end_node)
        new_relation.update(relation_properties)
        return new_relation

    def run_cypher_merge(self, cypher_batch):
        for i in range(len(cypher_batch)):
            cypher = cypher_batch[i].replace("CREATE", "MERGE").replace("create", "merge")
            cypher_batch[i] = cypher
        cypher = "\n".join(cypher_batch)
        try:
            ret = self.g.run(cypher).evaluate()
            print(f"Cypher done. Got ret {ret}")
        except Exception as e:
            print(f"{str(e)} occurs when executing {cypher}")

    def run_cypher(self, cypher_batch):
        cypher = "\n".join(cypher_batch)
        cyphers = cypher.strip().split(";")
        for c in cyphers:
            if c == "":
                continue
            try:
                ret = self.g.run(c).evaluate()
                # print(f"Cypher {c} done. Got ret {ret}")
            except Exception as e:
                print(c, str(e))

    @staticmethod
    def prep2dict(properties):
        prep_dict = {}
        splt = properties.split(",")
        for pair in splt:
            res = re.findall(r"(.*):\s*[\'\"](.*)[\'\"]", pair)
            if len(res) == 1:
                prep_dict[res[0][0].strip()] = res[0][1].strip()
        return prep_dict

    def getMsClassName(self):
        class_matcher = self.nodes_matcher.match("class").where(framework="pytorch")
        class_node = class_matcher.first()

    def getOpsName(self, fra):
        apiName = []
        api_matcher = self.nodes_matcher.match("operator").where(framework=fra)
        for api in api_matcher:
            apiName.append(api.get("full_name"))
        return apiName

    def getOperatorInfo(self, framework, api, version):
        """给定framework, api 查找 api 信息"""
        # 1. 获取算子节点
        api_matcher = self.nodes_matcher.match("operator").where(framework=framework, full_name=api, version=version)
        api_node = api_matcher.first()
        if api_node == None:
            return None, None, None, None

        # 2.1 获取算子父参数信息
        parameter = []
        params_macher = self.relationship_matcher.match((api_node, None), "parameterOfOperator")
        for params in params_macher:
            end = params.end_node
            p = json.dumps(end, default=lambda obj: obj.__dict__)
            parameter.append(p)
            # 3. 获取子参数信息
            child_params_macher = self.relationship_matcher.match((end, None), "oneOfParameter")
            for child_params in child_params_macher:
                pass
        # 2.2 得到排序后参数默认值
        orderedParams = []
        for param in parameter:
            jp = json.loads(param)
            orderedParams.append(Params(jp["name"], jp["parameter_order"], jp["dtype"], jp["default"]))
        orderedParams = sorted(orderedParams, key=attrgetter("order"))
        # print(parameter)

        # 3.1 获取算子父input信息
        input = []
        inputs_macher = self.relationship_matcher.match((api_node, None), "inputOfOperator")
        for inputs in inputs_macher:
            end = inputs.end_node
            p = json.dumps(end, default=lambda obj: obj.__dict__)
            input.append(p)

        # 3.2. 得到排序后参数默认值
        orderedInputs = []
        for i in input:
            jp = json.loads(i)
            orderedInputs.append(Params(jp["name"], jp["input_order"], jp["dtype"], jp["default"]))
        orderedInputs = sorted(orderedInputs, key=attrgetter("order"))

        # 3. 获取父返回值信息
        returns = []
        returns_macher = self.relationship_matcher.match((api_node, None), "returnOfOperator")
        for ret in returns_macher:
            end = ret.end_node
            # 3. 获取子返回信息
            child_returns_macher = self.relationship_matcher.match((end, None), "oneOfReturn")
            for child_return in child_returns_macher:
                child_end = child_return.end_node
                r = json.dumps(child_end, default=lambda obj: obj.__dict__)
                returns.append(r)
        return api_node, orderedParams, returns, orderedInputs

    def getMSInfoFromTorch(self, api):
        # 1. 获取算子节点
        version = "1.5.0"
        top, t_params, t_returns, _ = self.getOperatorInfo("pytorch", api, version)
        mindspore_matcher = self.relationship_matcher.match(
            (top, None), "equivalentOperator", framework_name="mindspore"
        )
        if top == None or mindspore_matcher.first() is None:
            version = "1.8.1"
            top, t_params, t_returns, _ = self.getOperatorInfo("pytorch", api, version)
            mindspore_matcher = self.relationship_matcher.match(
                (top, None), "equivalentOperator", framework_name="mindspore"
            )
            if top == None or mindspore_matcher.first() is None:
                # print("mindspore 中无 %s 对应算子" % api)
                return None, None, None
        ms_op = mindspore_matcher.first().end_node
        type_judgement = mindspore_matcher.first().get("type_judgement")
        ms_op, ms_params, ms_returns, _ = self.getOperatorInfo(
            "mindspore", ms_op.get("full_name"), ms_op.get("version")
        )

        # 2. 得到 ms 与 torch 参数对应关系
        ms_torch_relation = {"type_judgement": type_judgement}
        ms_params_matcher = self.relationship_matcher.match((ms_op, None), "parameterOfOperator")
        for ms_param_relation in ms_params_matcher:
            ms_param = ms_param_relation.end_node
            relation_macher = self.relationship_matcher.match((None, ms_param), "equivalentParameter")
            if relation_macher.first() is not None:
                torch_param = relation_macher.first().start_node
                ms_torch_relation[ms_param.get("name")] = torch_param.get("name")
        return ms_op, ms_params, ms_torch_relation


def process_cypher(path):
    txt_root = Path(path)
    for txt in txt_root.rglob("*.txt"):
        # print(f"extracting file {str(txt)}")
        with open(str(txt), "r", encoding="utf-8") as f:
            contents = f.readlines()
        kgraph.run_cypher(contents)


def main():
    # 存入图谱
    process_cypher("../../dao")


if __name__ == "__main__":
    kgraph = knowledgeGraph(clear=True)
    main()
