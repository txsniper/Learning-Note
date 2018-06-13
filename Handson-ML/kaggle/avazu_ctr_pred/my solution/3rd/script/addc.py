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


    def get_id(self, dev_id, dev_ip, dev_model):
        if dev_id != 'a99f214a':
            return dev_id
        else:
            return dev_ip + "_" + dev_model

    def process_data(self, input, output, isTest):
        f_out = open(output, "w")
        cols_name = []
        with open(input) as f_in:
            for line in f_in:
                line = line.strip()
                parts = line.split(",")
                if len(cols_name) == 0:
                    cols_name = parts
                    continue
                id = parts[0]
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

                id = self.get_id(device_id, device_ip, device_model)



                




    def run(self):
        self.process_data(self.train_in, self.train_out, False)
        self.process_data(self.test_in,  self.test_out,  True)

