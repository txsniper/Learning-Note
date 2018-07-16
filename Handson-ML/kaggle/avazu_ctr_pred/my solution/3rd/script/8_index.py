class Index(object):
    def __init__(self):
        pass

    def process(self, data_pre, fm_data_1, dic, isTest):
        f_in = open(data_pre)
        f_out = open(fm_data_1, "w")
        x = []
        y = []

        line = f_in.readline()
        idx = 3
        if isTest:
            idx = 2
        cnt = 0
        while True:
            line = f_in.readline().strip()
            if not line:
                break
            parts = line.split(",")
            if isTest:
                y.append(-1)
            else:
                label = int(parts[1])
                if label == 0:
                    label = -1
                y.append(label)
            
            cur_x = []
            for i in range(idx, len(parts)):
                if i == len(parts) - 2:
                    ss = parts[i].split('#')
                    if int(ss[1]) >= 50:
                        parts[i] = ss[0] + "#50"
                elif i == len(parts) - 5:
                    ss = parts[i].split("#")
                    if int(ss[i]) >= 20:
                        parts[i] = ss[0] + "#20"
                elif i > len(parts) - 8 and i < len(parts) - 1:
                    ss = parts[i].split("#")
                    if int(ss[1]) >= 10:
                        parts[i] = ss[0] + "#10"
                idx = dic.get(parts[i])
                if idx == None:
                    cur_x.append(len(dic))
                    d[parts[i]] = len(dic)
                else:
                    cur_x.append(idx)
            cur_str_x = [str(x) for x in cur_x]
            f_out.write(str(y[cnt]) + " " + " ".join(cur_str_x) + "\n")

            cnt = cnt + 1
            if cnt % 1000000 == 0:
                print("line count: " + str(cnt))
        f_in.close()
        f_out.close()
        return dic
                

    def run(self, train_pre, test_pre, fm_train_1, fm_test_1):
        dic = {}
        dic = self.process(train_pre, fm_train_1, dic, False)
        dic = self.process(test_pre, fm_test_1, dic, True)
 
    def process2(self, data_pre, fm_data_1_1, fm_data_1_2, isTest):
        f_in = open(data_pre)
        f_out_1 = open(fm_data_1_1, "w")
        f_out_2 = open(fm_data_1_2, "w")
        line = f_in.readline()
        idx = 3
        if isTest:
            idx = 2
        cnt = 0
        label = -1
        id = "??"
        dic = {}
        while True:
            isApp = False
            line = f_in.readline().strip()
            parts = line.split(",")
            id = parts[0]
            if isTest is False:
                label = int(parts[1])
                if label == 0:
                    label = -1
            cur_x = []

            # site_id
            if parts[idx+2] == "c#85f751fd":
                isApp = True
                dic = self.dic_app
            else:
                dic = self.dic_site
            for i in range(idx, len(parts)):
                if i == len(parts) - 2:
                    ss = parts[i].split("#")
                    if int(ss[1]) >= 50:
                        parts[i] = ss[0] + "#50"
                elif i == len(parts) - 5:
                    ss = parts[i].split("#")
                    if int(ss[i]) >= 20:
                        parts[i] = ss[0] + "#20"
                elif i > len(fields)-8 and i < len(fields)-1:
                ss = fields[i].split('#')
                if int(ss[1])>=10:
                    fields[i]=ss[0]+"#10"
            
                if isApp == True:
                    if fields[i][0] == "d" or fields[i][0] == "e" or fields[i][0] == "c":
                        continue
                else:
                    if fields[i][0] == "g" or fields[i][0] == "h" or fields[i][0] == "f": 
                        continue
                idx = dic.get(parts[i])
                if idx == None:
                    cur_x.append(len(dic))
                    dic[parts[i]] = len(dic)
                else:
                    cur_x.append(idx)
            cur_str_x = [str(x) for x in cur_x]
            if isApp is False:
                f_out_1.write(str(label) + " " + " ".join(cur_str_x))
            else:
                f_out_2.write(str(label) + " " + " ".join(cur_str_x))
            cnt = cnt + 1
            if cnt % 100000 == 0:
                print("line count: " + str(cnt))
        f_in.close()
        f_out_1.close()
        f_out_2.close()


    def run2(self, train_pre, test_pre, fm_train_1_1, fm_train_1_2, fm_test_1_1, fm_test_1_2):
        self.dic_app = {}
        self.dic_site = {}
        self.process2(train_pre, fm_train_1_1, fm_train_1_2, False)
        self.process2(test_pre, fm_test_1_1, fm_test_1_2, True)


if __name__ == "__main__":
    dir_in = "../data_out/"
    dir_out = "../data_out/"

    train_pre  = dir_in  + "train_pre"
    test_pre   = dir_in  + "test_pre"
    fm_train_1 = dir_out + "fm_train_1_1"
    fm_test_1  = dir_out + "fm_test_1_1"
    fm_train_2 = dir_out + "fm_train_1_2"
    fm_test_2  = dir_out + "fm_test_1_2"

    obj = Index()
    obj.run2(train_pre, test_pre, fm_train_1_1, fm_train_1_2, fm_test_1_1, fm_test_1_2)

