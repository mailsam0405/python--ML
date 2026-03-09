from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def marvellous_palypredi():
    #load dataset
    datasetpath="PlayPredictor.csv"
    df=pd.read_csv(datasetpath)
   
    print("dataset load succsefully")

    #label encoder
    le=LabelEncoder()

    df['Whether']=le.fit_transform(df['Whether'])
    df['Temperature']=le.fit_transform(df['Temperature'])
    df['Play']=le.fit_transform(df['Play'])

    #features and target
    
    X=df[['Whether','Temperature']]
    Y=df['Play']
    
    # split data
    X_train, X_test ,Y_train , Y_test=train_test_split(X,Y,test_size=0.2)

    #train data
    
    model=KNeighborsClassifier(n_neighbors=4)
    model.fit(X_train,Y_train)

    Y_pred=model.predict(X_test)

    
    
    #using user
    Whether=int(input("enter whether value(Overcast:0 Rainy:1 Sunny:2):"))
    temp=int(input("enter temp value(Cool:0 Hot:1 Mild:2):"))

    test_data=pd.DataFrame([[Whether,temp]],columns=['Whether','Temperature'])
    result=model.predict(test_data)

    if result[0]==1:
        print("Play : Yes")
    else:
        print("Play : No")
    
    accuracy=accuracy_score(Y_test,Y_pred)
    print("acuracy is :",accuracy*100)


def main():
    marvellous_palypredi()



if __name__=="__main__":
    main()