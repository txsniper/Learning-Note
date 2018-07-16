import marshal
from datetime import datetime
class GbdtDense(object):
    def __init__(self, train_in ,test_in, train_dense_out, test_dense_out):
        self.train_in = train_in
        self.test_in  = test_in
        self.train_dense_out = train_dense_out
        self.test_dense_out = test_dense_out

    def process(self, file_in, file_out, id_stat, isTest):
        f_in = open(file_in)
        f_out = open(file_out, "w")
        line = f_in.readline()
        idx = 3
        if isTest == True:
            idx = 2
        count = 0

        x = []
        y = []

        while True:
            line = f_in.readline().strip()
            if not line:
                break
            parts = line.split(",")
            if isTest == False:
                label = int(parts[1])
                if label == 0:
                    label = -1
                y.append(label)
            else:
                y.append(-1)

            cur_x = []
            for i in range(idx, len(parts)):
                # device_ip
                if i == len(parts) - 19:
                    cur_x.append(id_stat["j#" + parts[i]])
                # device_id
                elif i == len(parts) - 20:
                    continue
                # user id
                elif i == len(parts) - 7:
                    cur_x.append(id_stat["v#" + parts[i]])
                elif i > len(parts) - 7:
                    cur_x.append(int(parts[i]))

            cur_str_x = [str(x) for x in cur_x]
            line_out = str(y[count]) + " " + " ".join(cur_str_x)
            f_out.write(line_out + "\n")
            count = count + 1
            if count % 1000000 == 0:
                print("line count : " + str(count))
 

    def run(self, id_stat_file):
        start_time = datetime.now()
        id_stat = marshal.load(open(id_stat_file, "rb"))
        self.process(self.train_in, self.train_dense_out, id_stat, False)
        self.process(self.test_in, self.test_dense_out, id_stat, True)
        end_time = datetime.now()

if __name__ == "__main__":
    train_in = "./data_out/train_c"
    test_in  = "./data_out/test_c"

    train_out = "./data_out/train_dense"
    test_out = "./data_out/test_dense"

    id_stat_file = "./data_out/id_stat"
    obj = GbdtDense(train_in, test_in, train_out, test_out)
    obj.run(id_stat_file)
