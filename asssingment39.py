import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import(
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


Border="-"*40
################################################################################################
# Question 1 : Create a model object and train it using fit().
################################################################################################
print(Border)
print("Quetsion 1 : Create a model object and train it using fit().")
print(Border)

Datasetpath="student_performance_ml.csv"

df=pd.read_csv(Datasetpath)
X = df.drop("FinalResult", axis=1)
Y = df["FinalResult"]

# Step 2: Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


model=DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    random_state=42
)
print("model succsesfully created:",model)

model.fit(X_train,Y_train) 



################################################################################################
# Question 2 :use the train model to predict results for x_test.,disply predicted values along with actual values.
################################################################################################
print(Border)
print("Step 2 :use the train model to predict results for x_test.,disply predicted values along with actual values")
print(Border)


y_pred=model.predict(X_test)
result=pd.DataFrame({
     "Actual Value": Y_test.values,
    "Predicted Value": y_pred
    
})
print(result)


################################################################################################
# Question 3 : Calculate the model accuracy using accuracy_score,disply the result in percentage format
################################################################################################
print(Border)
print("Step 3 : Calculate the model accuracy using accuracy_score,disply the result in percentage format")
print(Border)

accuracy=accuracy_score(Y_test,y_pred)
print("accuracy of model is :",accuracy*100)


################################################################################################
# Question 4 : Generate confiusion matrix using slearn,display it using confusionmatrixdisplay
################################################################################################
print(Border)
print("Step 4 :Generate confiusion matrix using slearn,display it using confusionmatrixdisplay")
print(Border)

cm=confusion_matrix(Y_test,y_pred)
print(cm)
dispaly = ConfusionMatrixDisplay(confusion_matrix=cm)
dispaly.plot()

plt.show()

################################################################################################
# Question 5 :Calculate- Training Accuracy & Testing Accuracy
################################################################################################
print(Border)
print("Step 5 : Calculate- Training Accuracy & Testing Accuracy")
print(Border)

train_pred = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_pred)

print("Training accuracy of model is :", train_accuracy * 100)

test_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_pred)

print("Testing accuracy of model is :", test_accuracy * 100)




################################################################################################
#Question 6 : Train three Decision Treee models with-max_depth=1,3,none
################################################################################################
print(Border)
print("Step 6 : Train three Decision Treee models with-max_depth=1,3,none")
print(Border)

depth=[1,3,None]
for d in depth:
    model=DecisionTreeClassifier(max_depth=d)
    model.fit(X_train,Y_train)
    accuracy = model.score(X_test, Y_test)
    print("model with max_depth(", d," ) Accuracy:", accuracy)

################################################################################################
# Question 7 : Use the train model to predict result for a student 
################################################################################################
print(Border)
print(" Step 7 : Use the train model to predict result for a student ")
print(Border)

student=pd.DataFrame({

    "StudyHours": [6],
    "Attendance": [85],
    "PreviousScore": [65],
    "AssignmentsCompleted": [7],
    "SleepHours": [7]
})
predi=model.predict(student)
print("predicted result is :",predi)
if predi[0] == 1:
    print("Student will Pass")
else:
    print("Student will Fail")


################################################################################################
# Question 8 : create a single stucture python program
################################################################################################
print(Border)
print("Step 8 :  create a single stucture python program")
print(Border)

#step 1"dataset loading"
Datasetpath="student_performance_ml.csv"

df=pd.read_csv(Datasetpath)

print("dataset gets loaded succsefuly")
print("initial entries from data sets")
print(df.head())

#step 2 "data analysis"


print("shape of dataset:", df.shape)
print("column name:", list(df.columns))

print("missing values (per column)")
print(df.isnull().sum())

print("class distribution (FinalResult count)")
print(df["FinalResult"].value_counts())

print("statistical report of dataset")
print(df.describe())

#step 3 "define x or y variable"
feature_cols = [
    "StudyHours",
    "Attendance",
    "PreviousScore",
    "AssignmentsCompleted",
    "SleepHours"
]

X = df[feature_cols]
Y = df["FinalResult"]

print("X shape:", X.shape)
print("Y shape:", Y.shape)

#step 4 "visualization"
plt.figure(figsize=(6,4))
sns.scatterplot(x="PreviousScore", y="StudyHours",
                hue="FinalResult", data=df)
plt.title("Student Performance Visualization")
plt.grid(True)
plt.show()

# step 5 "train-test-split"
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

#step 6 "model training"

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

model.fit(X_train, Y_train)

print("Model Training Completed")

#step 7 "prediction"

Y_pred = model.predict(X_test)

#step 8 "accuracy calculation"
accuracy = accuracy_score(Y_test, Y_pred)

print("Testing Accuracy of Model is:", accuracy * 100)
 
#step 9 "confussion matrix"
cm = confusion_matrix(Y_test, Y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fail", "Pass"]
)

disp.plot()
plt.show()

# step 10 "final result"
student = pd.DataFrame({
    "StudyHours": [6],
    "Attendance": [85],
    "PreviousScore": [65],
    "AssignmentsCompleted": [7],
    "SleepHours": [7]
})

final_pred = model.predict(student)

if final_pred[0] == 1:
    print("Final Prediction: Student will Pass")
else:
    print("Final Prediction: Student will Fail") 


