import marshal
class IdStat(object):
    def __init__(self):
        pass
    
    def id_count(self, dic, parts, idx, prefix):
        id = prefix + parts[idx]
        if id in dic:
            dic[id] += 1
        else:
            dic[id] = 1
        return dic

    def process(self, file_in, dic, isTest):
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
                print("line count : " + str(count))
            parts = line.split(",")
            # id
            id_idx = len(parts) - 7
            dic = self.id_count(dic, parts, id_idx, "v#")
            # device_ip
            ip_idx = len(parts) - 19
            dic = self.id_count(dic, parts, ip_idx, "j#")
            # device_id
            did_idx = len(parts) - 20
            dic = self.id_count(dic, parts, did_idx, "i#")
        f_in.close()
        return dic

              
    def run(self, train_in, test_in, file_out):
        dic = {}
        dic = self.process(train_in, dic, False)
        dic = self.process(test_in, dic, True)
        marshal.dump(dic, open(file_out, "wb"))

if __name__ == "__main__":
    train_in = "./data_out/train_c"
    test_in  = "./data_out/test_c"
    file_out = "./data_out/id_stat"
    obj = IdStat()
    obj.run(train_in, test_in, file_out)
