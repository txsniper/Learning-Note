import marshal

class IdDay(object):
    def __init__(self, train_in, test_in, file_out):
        self.train_in = train_in
        self.test_in = test_in
        self.file_out = file_out

    def run(self):
        dic = {}
        dic = self.process_file(self.train_in, dic, False)
        dic = self.process_file(self.test_in, dic, True)

        d_set = {}
        for k in dic:
            d_set[k] = len(dic[k])
        
        marshal.dump(d_set, open(self.file_out, "w"))


    def process_file(self, file_in, dic, isTest):
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

            # ip
            ip_idx = 12
            date_idx = 2
            if isTest:
                ip_idx = 11
                date_idx = 1

            ip = "j#" + parts[ip_idx]
            date = parts[date_idx]
            day = date[4:6]
            if ip in dic:
                dic[ip].add(day)   
            else:
                s = set()
                s.add(day)
                dic[ip] = s

            iid_idx = len(parts) - 7
            iid = "v#" + parts[iid_idx]
            if iid in dic:
                dic[iid].add(day)
            else:
                s = set()
                s.add(day)
                dic[iid] = s
        f_in.close()     


if __name__ == "__main__":
    train_in = "./data_out/train_c"
    test_in = "./data_out/test_c"
    file_out = "./data_out/id_day"
    obj = IdDay(train_in, test_in, file_out)
    obj.run()

