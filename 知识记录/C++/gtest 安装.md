## 安装 gtest (非系统目录)
1. 下载 googletest 安装包并解压
2. 进入 googletest目录并执行(假设 GTEST_DIR 就是googletest源码目录):
```
g++ -std=c++11 -isystem ${GTEST_DIR}/include -I${GTEST_DIR} \
    -pthread -c ${GTEST_DIR}/src/gtest-all.cc
ar -rv libgtest.a gtest-all.o
```
3. 因为没有将gtest的include和libgtest.a 放到系统目录下，因此在使用 gtest 时需要为g++ 指定 gtest 头文件目录和 为 ld指定 libgtest.a的路径，下面以编译 xgboost 为例, -I 指定头文件目录，-L指定链接目录 (在编译xgboost时，直接在 make/config.mk 中指定 gtest 目录)
```
g++ -DDMLC_LOG_CUSTOMIZE=1 -std=c++11 -Wall -Wno-unknown-pragmas -Iinclude   -Idmlc-core/include -Irabit/include -I~/download/googletest-master/googletest//include -g -O0 -fprofile-arcs -ftest-coverage -fPIC -fopenmp -I~/download/googletest-master/googletest//include/ -o build_tests/cpp/c_api/test_c_api.o -c tests/cpp/c_api/test_c_api.cc

g++ -DDMLC_LOG_CUSTOMIZE=1 -std=c++11 -Wall -Wno-unknown-pragmas -Iinclude   -Idmlc-core/include -Irabit/include -I~/download/googletest-master/googletest//include -g -O0 -fprofile-arcs -ftest-coverage -fPIC -fopenmp -o tests/cpp/xgboost_test  build_tests/cpp/c_api/test_c_api.o lib/libxgboost.a dmlc-core/libdmlc.a rabit/lib/librabit.a -pthread -lm  -fopenmp -lrt  -lrt -L~/download/googletest-master/googletest//lib/ -lgtest
```