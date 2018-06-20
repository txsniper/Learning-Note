cd ../fm
make
cd ../gbdt
make

cd ../script

# step1: 增加8个额外特征
pypy addc.py
# step2: 统计特征次数，获取频次大于等于10的特征
pypy fcount.py
# step3: 统计出现次数小于10的id和ip
pypy rare.py