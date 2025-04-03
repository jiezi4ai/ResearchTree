import json
from neo4j import GraphDatabase  # pip install neo4j https://github.com/neo4j/neo4j-python-driver


def is_neo4j_compatible(value):
    """check if the value compatible with neo4j data types"""
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    elif isinstance(value, list):
        return all(is_neo4j_compatible(item) for item in value)
    else:
        return False
        

class JsonNeo4jPortal:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

    def import_json_to_neo4j(self, processed_json_data, neo4j_database):
        """import json to neo4j
        Note:
            - the json has to be preprocessed to in format like:
                [{'type': 'node',
                'id': '2345003971',
                'labels': ['Author'],
                'properties': {'authorId': '2345003971', 'name': 'Mark Schone'}},
                {'type': 'relationship',
                'relationshipType': 'WRITES',
                'startNodeId': '2345003971',
                'endNodeId': '10.48550/arXiv.2502.07827',
                'properties': {'authorOrder': 1,
                'coauthors': [{'authorId': '2345003971', 'name': 'Mark Schone'},]}}]
            - if there is multiple layers of dicts in properties, it would be converted into json format
        """
        driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user , self.neo4j_password ))

        with driver.session(database=neo4j_database) as session:
            for item in processed_json_data:
                if item['type'] == 'node':
                    labels = ":".join(item['labels'])
                    parameters = {"id": item['id']}
                    set_clauses = []

                    if item.get('properties') and isinstance(item['properties'], dict):
                        for key, value in item['properties'].items():
                            if is_neo4j_compatible(value):
                                parameters[key] = value
                            else:
                                # 序列化非兼容类型为JSON字符串
                                parameters[key] = json.dumps(value, ensure_ascii=False)
                            set_clauses.append(f"n.{key} = ${key}")

                    merge_query = f"MERGE (n:{labels} {{id: $id}})"
                    if set_clauses:
                        set_query = "SET " + ", ".join(set_clauses)
                        cypher_query = f"""
                            {merge_query}
                            ON CREATE {set_query}
                            ON MATCH {set_query}
                        """
                    else:
                        cypher_query = merge_query
                    cypher_query += " RETURN n"
                    session.run(cypher_query, parameters)

                elif item['type'] == 'relationship':
                    rel_type = item['relationshipType']
                    parameters = {"startId": item['startNodeId'], "endId": item['endNodeId']}
                    set_clauses = []

                    if item.get('properties') and isinstance(item['properties'], dict):
                        for key, value in item['properties'].items():
                            if is_neo4j_compatible(value):
                                parameters[key] = value
                            else:
                                parameters[key] = json.dumps(value, ensure_ascii=False)
                            set_clauses.append(f"r.{key} = ${key}")

                    cypher_query = f"""
                        MATCH (a {{id: $startId}}), (b {{id: $endId}})
                        MERGE (a)-[r:{rel_type}]->(b)
                    """
                    if set_clauses:
                        set_query = "SET " + ", ".join(set_clauses)
                        cypher_query += f"""
                            ON CREATE {set_query}
                            ON MATCH {set_query}
                        """
                    cypher_query += " RETURN r"
                    session.run(cypher_query, parameters)
        driver.close()

    def _batch_import_nodes(self, tx, node_batch):
        """nodes batch import
        """
        cypher_query = """
        UNWIND $batch AS item
        MERGE (n:Node {id: item.id}) // 使用一个通用的 Node 标签作为初始标签，后续再添加具体标签
        ON CREATE SET n = item.properties, n += {id: item.id}
        ON MATCH SET n = item.properties, n += {id: item.id}
        SET n:Node // 确保 Node 标签始终存在
        FOREACH (label IN item.labels | SET n:Label) // 动态添加 labels
        """
        # 准备批量参数，properties 需要处理成 Neo4j 兼容的格式
        batch_parameters = []
        for item in node_batch:
            parameters = {"id": item['id'], "labels": item['labels'], "properties": {}}
            if item.get('properties') and isinstance(item['properties'], dict):
                for key, value in item['properties'].items():
                    if is_neo4j_compatible(value):
                        parameters["properties"][key] = value
                    else:
                        parameters["properties"][key] = json.dumps(value, ensure_ascii=False) # 序列化非兼容类型
            batch_parameters.append(parameters)
        tx.run(cypher_query, {"batch": batch_parameters})

    def _batch_import_relationships(self, tx, relationship_batch):
        """批量导入关系的事务函数"""
        cypher_query = """
        UNWIND $batch AS item
        MATCH (a:Node {id: item.startNodeId}), (b:Node {id: item.endNodeId})
        MERGE (a)-[r:Relationship]->(b) // 使用一个通用的 Relationship 类型，后续再设置具体类型
        ON CREATE SET r = item.properties
        ON MATCH SET r = item.properties
        SET TYPE(r) = item.relationshipType // 动态设置关系类型
        """
        batch_parameters = []
        for item in relationship_batch:
            parameters = {"startNodeId": item['startNodeId'], "endNodeId": item['endNodeId'],
                        "relationshipType": item['relationshipType'], "properties": {}}
            if item.get('properties') and isinstance(item['properties'], dict):
                for key, value in item['properties'].items():
                    if is_neo4j_compatible(value):
                        parameters["properties"][key] = value
                    else:
                        parameters["properties"][key] = json.dumps(value, ensure_ascii=False) # 序列化非兼容类型
            batch_parameters.append(parameters)
        tx.run(cypher_query, {"batch": batch_parameters})

    def batch_import_json_to_neo4j(self, processed_json_data, neo4j_database, batch_size=100):
        driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))

        node_batches = []
        relationship_batches = []
        current_node_batch = []
        current_relationship_batch = []

        for item in processed_json_data:
            if item['type'] == 'node':
                current_node_batch.append(item)
                if len(current_node_batch) >= batch_size:
                    node_batches.append(current_node_batch)
                    current_node_batch = []
            elif item['type'] == 'relationship':
                current_relationship_batch.append(item)
                if len(current_relationship_batch) >= batch_size:
                    relationship_batches.append(current_relationship_batch)
                    current_relationship_batch = []

        # 处理剩余的未满 batch_size 的数据
        if current_node_batch:
            node_batches.append(current_node_batch)
        if current_relationship_batch:
            relationship_batches.append(current_relationship_batch)

        with driver.session(database=neo4j_database) as session:
            # 批量导入节点
            for batch in node_batches:
                tx = session.begin_transaction()
                try:
                    self._batch_import_nodes(tx, batch)
                    tx.commit()
                except Exception as e:
                    tx.rollback()
                    print(f"节点批量导入事务回滚: {e}")

            # 批量导入关系
            for batch in relationship_batches:
                tx = session.begin_transaction()
                try:
                    self._batch_import_relationships(tx, batch)
                    tx.commit()
                except Exception as e:
                    tx.rollback()
                    print(f"关系批量导入事务回滚: {e}")

        driver.close()