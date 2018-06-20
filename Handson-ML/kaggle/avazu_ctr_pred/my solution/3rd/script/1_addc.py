from collections import defaultdict
import marshal

class Addc(object):
    def __init__(self, train_in, test_in, train_out, test_out):
        self.train_in   = train_in
        self.test_in    = test_in
        self.train_out  = train_out
        self.test_out   = test_out
        self.col_idx = {
            'hour' : 0
            'C1' : 1,
            'banner_pos' : 2,
            'site_id' : 3,
            'site_domain' : 4,
            'site_category' : 5,
            'app_id' : 6,
            'app_domain' : 7,
            'app_category' : 8,
            'device_id' : 9,
            'device_ip' : 10,
            'device_model' : 11,
            'device_type' : 12,
            'device_conn_type' : 13,
            'C14' : 14,
            'C15' : 15,
            'C16' : 16,
            'C17' : 17,
            'C18' : 18,
            'C19' : 19,
            'C20' : 20,
            'C21' : 21,
        }
        self.idx_col = {}
        for key, value in self.col_idx:
            self.idx_col[value] = key

        self.feature_prefix = {
            "site_id":    "c",
            'app_id' :    "f",
            'device_id' : "i",
            "device_ip" : "j",
            "device_model" : "k",
            'device_type' : "l",
            'device_conn_type' : "m",
            'C14' : "n",
            'C15' : "o",
            'C16' : "p",
            'C17' : "q",
            'C18' : "r",
            'C19' : "s",
            'C20' : "t",
            'C21' : "u",  
        }

    # 利用device_id来标识用户，如果device_id是"a99f214a"，
    # 则使用 dev_ip + dev_model 组合
    def get_id(self, dev_id, dev_ip, dev_model):
        if dev_id != 'a99f214a':
            return dev_id
        else:
            return dev_ip + "_" + dev_model


    def process_data(self, input, output, isTest):
        f_out = open(output, "w")
        
        cols_name = []
        line_count = 0

        # 按天编号: 用户id与广告id的组合
        id_day_impression = defaultdict(int)

        # 按小时编号： 用户id与广告id的组合
        id_hour_impression = defaultdict(int)

        # 同一天内访问的时间间隔
        id_last_visit_time_interval = {}

        last_day = "??"
        last_hour = "??"

        with open(input) as f_in:
            for line in f_in:
                line = line.strip()
                if len(line) == 0:
                    break
                line_count += 1
                if line_count == 1:
                    f_out.write(line[:-2] + ",C22,C23,C24,C25,C26,C27,C28\n")
                if line_count % 100000 == 0:
                    print("line count : " + str(line_count))
                parts = line.split(",")
                if len(cols_name) == 0:
                    cols_name = parts
                    continue
                id = parts[0]

                # 训练数据和测试数据的 date_idx 不一样
                date_idx = 2
                if isTest is True:
                    date_idx = 1
                
                # 14102100  --> 2014,10:21:00
                date            = parts[date_idx + 0]
                C1              = parts[date_idx + 1]
                banner_pos      = parts[date_idx + 2]
                site_id         = parts[date_idx + 3]
                site_domain     = parts[date_idx + 4]
                site_category   = parts[date_idx + 5]
                app_id          = parts[date_idx + 6]
                app_domain      = parts[date_idx + 7]
                app_category    = parts[date_idx + 8]
                device_id       = parts[date_idx + 9]
                device_ip       = parts[date_idx + 10]
                device_model    = parts[date_idx + 11]
                device_type     = parts[date_idx + 12]
                device_conn_type = parts[date_idx + 13]
                C14             = parts[date_idx + 14]
                C15             = parts[date_idx + 15]
                C16             = parts[date_idx + 16]
                C17             = parts[date_idx + 17]
                C18             = parts[date_idx + 18]
                C19             = parts[date_idx + 19]
                C20             = parts[date_idx + 20]
                C21             = parts[date_idx + 21]

                year  = date[:2]
                month = date[2:4]
                day   = date[4:6]
                hour = date[6:]

                # 从新的一天开始编号
                if day != last_day:
                    id_day_impression = defaultdict(int)
                    last_day = day
                    id_last_visit_time_interval = {}
                
                # 从新的一小时开始编号
                if hour != last_hour:
                    id_hour_impression = defaultdict(int)
                    last_hour = hour
                
                curr_time = int(hour) * 60

                id = self.get_id("i#" + device_id, "j#" + device_ip, "k#" + device_model)
                id_C14 = id + "_" + "n#" + C14
                id_C17 = id + "_" + "q#" + C17

                # 按天编号
                id_day_impression[id_C14] += 1
                id_day_impression[id_C17] += 1

                # 按小时编号
                id_hour_impression[id_C14] += 1
                id_hour_impression[id_C17] += 1
                id_hour_impression[id] += 1

                median_id = "f#" + app_id
                if app_id == "ecad2386":
                    median_id = "c#" + site_id
                
                id_day_impression[id + "_" + median_id] += 1

                t = "-1"
                if id not in id_last_visit_time_interval:
                    id_last_visit_time_interval[id] = curr_time
                else:
                    t = str(curr_time - id_last_visit_time_interval[id])
                    id_last_visit_time_interval[id] = curr_time

                m  = id_day_impression[id + "_" + median_id]
                c  = id_day_impression[id_C14]
                c2 = id_day_impression[id_C17]

                ch  = id_hour_impression[id_C14]
                ch1 = id_hour_impression[id_C17]
                ch2 = id_hour_impression[id]

                out_line = line[:-2] + "," + id + "," + str(m) + "," + str(ch1) + "," + str(ch2) + "," + str(c) + "," + str(c2) + "," + t
                f_out.write(out_line + "\n")
        f_out.close()

        
    def run(self):
        self.process_data(self.train_in, self.train_out, False)
        self.process_data(self.test_in,  self.test_out,  True)


if __name__ == "__main__":
    train_in = "./data_in/train"
    test_in = "./data_in/test"
    train_out = "./data_out/train_c"
    test_out = "./data_out/test_c"
    obj = Addc(train_in, test_in, train_out, test_out)
    obj.run()