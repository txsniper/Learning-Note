cd ../fm
make
cd ../gbdt
make

cd ../script

# step1: 增加8个额外特征
pypy 1_addc.py
# step2: 统计特征次数，获取频次大于等于10的特征
pypy 2_fcount.py
# step3: 统计出现次数小于10的id和ip
pypy 3_rare.py
# step4: id/ip出现的天数
pypy 4_id_day.py
# step5: 替换出现次数太少的id为定值
pypy 5_prep.py
# step6: 统计id, ip, 出现的次数
pypy 6_id_stat.py
# step7: 生成gbdt的特征
pypy 7_gbdt_dense.py
