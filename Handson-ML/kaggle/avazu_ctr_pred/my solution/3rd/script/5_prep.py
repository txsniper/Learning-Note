import marshal

class Prep(object):
    def __init__(self):
        pass
    def process(self, train_in, test_in, train_out, test_out, fc_file, rare_file, id_day_file):
        fc_in = marshal.load(open(fc_file))
        rare_in = marshal.load(open(rare_file))
        id_day_in = marshal.load(open(id_day_file))
        self.process_one_file(fc_in, rare_in, id_day_in, train_in, train_out, False)
        self.process_one_file(fc_in, rare_in, id_day_in, test_in, test_out, True)
    
    def get_rare_data(self, prefix, rare_in, parts, idx):
        id = parts[idx]
        id = prefix + "#" + id
        rare = rare_in.get(id)
        if rare != None:
            return prefix + "_rare_" + rare
        return None

    def process_one_file(self, fc_in, rare_in, id_day_in, file_in, file_out, isTest):
        f_in = open(file_in)
        f_out = open(file_out, "w")
        line = f_in.readline()
        f_out.write(line[:-1] + "\n")
        count = 1
        C1_idx = 3
        if isTest:
            C1_idx = 2
        while True:
            line = f_in.readline()
            line = line.strip()
            if not line:
                break
            count += 1
            if count % 100000 == 0:
                print("line count : " + str(count))
            parts = line.split(",")
            uid = "???
            for i in range(C1_idx, len(parts)):
                prefix = chr(ord('a') + i - C1_idx)

                # device_ip
                if prefix == "j":
                    parts[i] = self.get_rare_data(prefix, parts[i], rare_in)
                    ip = prefix + "#" + parts[i]
                    rare = rare_in.get(ip)
                    if rare != None:
                        parts[i] = "j#rare#" + str(rare)
                        continue
                
                # device_id
                if prefix == "i":
                    id = prefix + "#" + parts[i]
                    rare = rare_in.get(id)
                    if rare != None:
                        parts[i] = "i#rare#" + str(rare)
                        continue

                # addc中用户的复合id
                if prefix == 'v':
                    id = prefix + "#" + parts[i]
                    rare = rare_in.get(id)
                    if rare != None:
                        parts[i] = "v_rare_" + str(rare)
                        continue
                    elif id_day_in.get(id) == 1:
                        parts[i] = "v_id_s"
                        continue

                # 对于其他的次数太少的特征直接用定值代替
                if prefix + "#" + parts[i] not in fc_in and i < len(parts) - 6:
                    parts[i] = prefix + "_rare"
                else:
                    parts[i] = prefix + "#" + parts[i]
            parts.append("id_day_" + str(id_day_in[id]))
            f_out.write(",".join(parts) + "\n")
        f_in.close()
        f_out.close()


if __name__ == "__main__":
    train_in = "./data_out/train_c"
    test_in = "./data_out/test_c"
    fcount_in = "./data_out/fc"
    rare_in = "./data_out/rare_d"
    id_day_in = "./data_out/id_day"
    train_out = "./data_out/train_pre"
    test_out = "./data_out/test_pre"
    prep = Prep()
    prep.process(train_in, test_in, train_out, test_out, fcount_in, rare_in, id_day_in)
