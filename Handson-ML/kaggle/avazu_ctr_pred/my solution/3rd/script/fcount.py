import marshal
class Fcount(object):
    def __init__(self, train_in, test_in, file_out):
        self.train_in = train_in
        self.test_in = test_in
        self.file_out = file_out

    def process_file(self, file_in, dic, is_test):
        f_in = open(file_in)
        first_line = f_in.readline()

        if is_test:
            start_pos = 2
        else:
            start_pos = 3
        count = 1
        while True:
            line = f_in.readline()
            line = line.strip()
            if not line:
                break
            count += 1
            if count % 100000 == 0:
                print("line count : " + str(count))
            
            parts = line[:-2].split(",")
            for i in range(start_pos, len(parts)):
                prefix = chr(ord('a') + i - 3)
                feature = prefix + "#" + parts[i]
                if feature in dic:
                    dic[feature] += 1
                else:
                    dic[feature] = 1
        return dic
        
    def run(self):
        dic = {}
        dic = self.process_file(self.train_in, dic, False)
        dic = self.process_file(self.test_in, dic, True)
        f_out = open(self.file_out, "w")

        s = []
        for feature in dic:
            if dic[feature] >= 10:
                s.append(feature)
        marshal.dump(set(s), f_out)


if __name__ == "__main__":
    train_in = "./data_in/train"
    test_in = "./data_in/test"
    file_out = "./data_out/fc"
    obj = Fcount(train_in, test_in, file_out)
    obj.run()

        