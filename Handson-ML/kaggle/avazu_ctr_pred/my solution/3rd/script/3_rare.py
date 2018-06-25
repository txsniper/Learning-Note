import marshal
from collections import defaultdict

class Rare(object):
    def __init__(self, train_in, test_in, file_out):
        self.train_in = train_in
        self.test_in = test_in
        self.file_out = file_out

    def process_file(self, file_in, dic, is_test):
        f_in = open(file_in)
        line = f_in.readline()
        count = 1
        while True:
            line = f_in.readline()
            line = line.strip()
            if not line:
                break
            count += 1
            if count % 100000 == 0:
                print("line count: " + str(count))
            parts = line.split(",")
            device_id_idx = 11
            if is_test:
                device_id_idx = 10

            id = "i#" + parts[device_id_idx]
            # device_ip
            ip = "j#" + parts[device_id_idx + 1]

            # addr中生成的id
            iid = "v#" + parts[len(parts) - 7]
            dic[id] += 1
            dic[ip] += 1
            dic[iid] += 1
        f_in.close()
        return dic
    
    def run(self):
        dic = defaultdict(int)
        dic = self.process_file(self.train_in, dic, False)
        dic = self.process_file(self.test_in, dic, True)
        rare_d = {}
        for id in dic:
            if dic[id] <= 10:
                rare_d[id] = dic[id]

        marshal.dump(rare_d, open(self.file_out, "wb"))


if __name__ == "__main__":
    train_in = "./data_out/train_c"
    test_in = "./data_out/test_c"
    file_out = "./data_out/rare_d"
    obj = Rare(train_in, test_in, file_out)
    obj.run()


