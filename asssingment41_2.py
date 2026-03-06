#predict class of the same  nwe point using: 
# k=1 
# k=3 
# k=5


import math

def eucdistance(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)



def MarvellousKNN():

    border="-"*40

    data=[
        {'point':'A','X':1,'Y':2,'label':'Red'},
        {'point':'B','X':2,'Y':3,'label':'Red'},
        {'point':'C','X':3,'Y':1,'label':'Blue'},
        {'point':'D','X':5,'Y':6,'label':'Blue'}
    ]

    print(border)
    print("Training Dataset")
    print(border)

    for d in data:
        print(d)

    print(border)

    # Accept input
    x=int(input("Enter X coordinate: "))
    y=int(input("Enter Y coordinate: "))

    new_point={'X':x,'Y':y}

    # Calculate distance
    for d in data:
        d['distance']=eucdistance(d['X'],d['Y'],new_point['X'],new_point['Y'])

    # Sort distances
    sorted_data=sorted(data,key=lambda item:item['distance'])

    # Select K neighbors
    for K in [1,3,5]:
        neighbors=sorted_data[:K]

        print(border)
        print("Nearest Neighbors:")
        print(border)

        for n in neighbors:
             print(n['point'],"- Distance:",round(n['distance'],2))

    # Majority voting
        red=0
        blue=0

        for n in neighbors:
            if n['label']=="Red":
                red+=1
            else:
                blue+=1

            print(border)

        if red>blue:
            print("Predicted Class: Red")
        else:
            print("Predicted Class: Blue")

        print(border)


def main():
    MarvellousKNN()


if __name__=="__main__":
    main()


