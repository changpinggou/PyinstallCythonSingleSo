import json
import copy
import os


class JSONDB:
    def __init__(self, json_file):
        self.json_file = json_file

        # 如果JSON文件不存在，则创建一个空的JSON文件
        if not os.path.exists(json_file):
            os.makedirs(os.path.dirname(json_file))
            with open(json_file, 'w') as f:
                json.dump({}, f)
                f.close()

        with open(json_file, 'r') as f:
            self.data = json.load(f)
            f.close()

    def save(self):
        with open(self.json_file, 'w') as f:
            json.dump(self.data, f, indent=4)
            f.close()

    

    def save_item(self, value):
        #key = str(uuid.uuid4())
        self.data[value.id] = value
        self.save()
        return copy.deepcopy(value)

    def insert(self, key, value):
        #key = str(uuid.uuid4())
        value['id'] = key
        self.data[key] = value
        self.save()
        return copy.deepcopy(value)

    def delete(self, key):
        del self.data[key]
        self.save()

    def get(self, key):
        return copy.deepcopy(self.data.get(key))

    def get_all(self):
        return copy.deepcopy(self.data).values()
