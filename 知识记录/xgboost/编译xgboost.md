# 编译 xgboost
普通编译直接按照 xgboost 提供的文档就行，如果想编译 xgboost 中的 test case，需要首先安装 gtest， 然后在 make/config.mk中指定 GTEST_PATH (同时需要修改标志:TEST_COVER or BUILD_TEST)