import pandas as pd
def main():
    path = ""
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

if __name__ == "__main__":
    main()