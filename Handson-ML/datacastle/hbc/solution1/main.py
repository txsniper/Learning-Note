import pandas as pd
import time
import math

def encode_onehot(df,column_name):
    feature_df=pd.get_dummies(df[column_name], prefix=column_name)
    all = pd.concat([df.drop([column_name], axis=1),feature_df], axis=1)
    return all

def encode_count(df,column_name):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df[column_name].values))
    df[column_name] = lbl.transform(list(df[column_name].values))
    return df

def get_date(timestamp):
    time_local = time.localtime(timestamp)
    #dt = time.strftime("%Y-%m-%d %H",time_local)
    dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
    return dt

def handleOrderHistory(order_history):
    order_history["date"]=order_history["orderTime"].apply(get_date)
    order_history["date"]=pd.to_datetime(order_history["date"])
    order_history["weekday"]=order_history["date"].dt.weekday
    order_history["hour"]=order_history["date"].dt.hour
    order_history["month"]=order_history["date"].dt.month
    order_history["day"]=order_history["date"].dt.day
    order_history["minute"]=order_history["date"].dt.minute
    order_history["second"]=order_history["date"].dt.second
    order_history['tm_hour']=order_history['hour']+order_history['minute']/60.0
    order_history['tm_hour_sin'] = order_history['tm_hour'].map(lambda x: math.sin((x-12)/24*2*math.pi))
    order_history['tm_hour_cos'] = order_history['tm_hour'].map(lambda x: math.cos((x-12)/24*2*math.pi))
    return order_history


def handleAction(action, order_history_action):
    order_history_action["actionType"]=order_history_action["actionType"].apply(lambda x:x+10)
    action=pd.concat([action,order_history_action])
    action["date"]=action["actionTime"].apply(get_date)
    action["date"]=pd.to_datetime(action["date"])
    action["weekday"]=action["date"].dt.weekday
    action["hour"]=action["date"].dt.hour
    action["month"]=action["date"].dt.month
    action["day"]=action["date"].dt.day
    action["minute"]=action["date"].dt.minute
    action["second"]=action["date"].dt.second
    action['tm_hour']=action['hour']+action['minute']/60.0
    action['tm_hour_sin'] = action['tm_hour'].map(lambda x: math.sin((x-12)/24*2*math.pi))
    action['tm_hour_cos'] = action['tm_hour'].map(lambda x: math.cos((x-12)/24*2*math.pi))

    # 先按userid，再按actionTime排序
    action=action.sort_values(["userid","actionTime"])
    
    action["date"]=action["actionTime"].apply(get_date)

    #print(action['actionTime'].shift(1))
    #计算上下两条在指定列之间的差值
    action["actionTime_gap"]=action["actionTime"]-action["actionTime"].shift(1)
    action["actionType_gap"]=action["actionType"].shift(1)-action["actionType"]
    action["actionTime_long"]=action["actionTime"].shift(-1)-action["actionTime"]
    action["actionTime_gap_2"]=action["actionTime_gap"]-action["actionTime_gap"].shift(1)
    action["actionTime_long_2"]=action["actionTime_long"]-action["actionTime_long"].shift(1)
    #print(action[0:5])
    
    return action

def getOrderHistoryTime(order_history_time, action):
    this_time=action.drop_duplicates("userid",keep="last")[["userid","actionTime"]].copy()
    this_time.columns=["userid","orderTime"]
    order_history_time=pd.concat([order_history_time,this_time])
    order_history_time=order_history_time.drop_duplicates()
    order_history_time=order_history_time.sort_values(["userid","orderTime"])
    order_history_time["orderTime_gap"]=order_history_time["orderTime"]-order_history_time["orderTime"].shift(1)
    return order_history_time

def getUserProfile(user_profile):
    user_profile=encode_count(user_profile,"gender")
    user_profile=encode_count(user_profile,"province")
    user_profile=encode_count(user_profile,"age")
    return user_profile

def main():
    path = "../input/"
    userProfile_train=pd.read_csv(path+"userProfile_train.csv")
    userProfile_test=pd.read_csv(path+"userProfile_test.csv")
    userComment_train=pd.read_csv(path+"userComment_train.csv")
    userComment_test=pd.read_csv(path+"userComment_test.csv")
    orderHistory_train=pd.read_csv(path+"orderHistory_train.csv")
    orderHistory_test=pd.read_csv(path+"orderHistory_test.csv")
    orderFuture_train=pd.read_csv(path+"orderFuture_train.csv")
    orderFuture_test=pd.read_csv(path+"orderFuture_test.csv")
    action_train=pd.read_csv(path+"action_train.csv")
    action_test=pd.read_csv(path+"action_test.csv") 


    
    orderFuture_test["orderType"]=-1
    data=pd.concat([orderFuture_train,orderFuture_test])
    
    user_comment=pd.concat([userComment_train,userComment_test])

    # 处理历史订单
    order_history=pd.concat([orderHistory_train,orderHistory_test])
    order_history = handleOrderHistory(order_history)
    
    
    action=pd.concat([action_train,action_test])
    #处理操作日志
    # 从history中提取和action日志相同的字段，然后拼接在一起
    order_history_action=order_history[["userid","orderTime","orderType"]].copy()
    order_history_action.columns=["userid","actionTime","actionTyaction
    action = handleAction(action, order_history_action)

    order_history_time=order_history[["userid","orderTime"]].copy()
    order_history_time = getOrderHistoryTime(order_history_time, action)


    # 处理用户信息
    user_profile=pd.concat([userProfile_train,userProfile_test]).fillna(-1)
    user_profile = getUserProfile(user_profile)
    data=data.merge(user_profile,on="userid",how="left")
    

if __name__ == "__main__":
    main()