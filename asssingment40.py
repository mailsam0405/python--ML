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


df = pd.read_csv("student_performance_ml.csv")

# Define features and target
features = ["StudyHours", "Attendance", "PreviousScore", "AssignmentsCompleted", "SleepHours"]
X = df[features]
Y = df["FinalResult"]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the Decision Tree model
model = DecisionTreeClassifier(
    criterion="gini", 
    max_depth=3, 
    random_state=42
    
)


model.fit(X_train, Y_train)

# Feature importance
importance = model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    "features": feature_names,
    "importance": importance
}).sort_values(by="importance", ascending=False)

print(Border)
print("Step 12 : Feature Importance")
print(Border)
print(feature_importance_df)

################################################################################################
# Question 2 :remove the column sleephours from dataset
################################################################################################
print(Border)
print("Step 2 :remove the column sleephours from dataset")
print(Border)
X_old = df.drop("FinalResult", axis=1)
Y_old = df["FinalResult"]

X_train_old, X_test_old, Y_train_old, Y_test_old = train_test_split(
    X_old, Y_old, test_size=0.3, random_state=42
)

model_old = DecisionTreeClassifier()
model_old.fit(X_train_old, Y_train_old)

Y_pred_old = model_old.predict(X_test_old)

old_accuracy = sum(Y_test_old == Y_pred_old) / len(Y_test_old)

print("Previous Accuracy (With SleepHours) :", old_accuracy * 100)



# NEW MODEL 


X_new = df.drop(["FinalResult", "SleepHours"], axis=1)
Y_new = df["FinalResult"]

X_train_new, X_test_new, Y_train_new, Y_test_new = train_test_split(
    X_new, Y_new, test_size=0.3, random_state=42
)

model_new = DecisionTreeClassifier()
model_new.fit(X_train_new, Y_train_new)

Y_pred_new = model_new.predict(X_test_new)

new_accuracy = sum(Y_test_new == Y_pred_new) / len(Y_test_new)

print("New Accuracy (Without SleepHours) :", new_accuracy * 100)



# Compare Accuracy


difference = (new_accuracy - old_accuracy) * 100

print("Difference in Accuracy :", difference)

################################################################################################
# Question 3 : Train the model only
################################################################################################
print(Border)
print("Step 3 : Train the model only ")
print(Border)

feature=["StudyHours","Attendance"]
X=df[feature]
Y=df["FinalResult"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

model.fit(X_train, Y_train)


print("Model Training Completed")


################################################################################################
# Question 4 :create new dataframe with details 5 new students
################################################################################################
print(Border)
print("Step 4 :create new dataframe with details 5 new students")
print(Border)

#create database
new_students=pd.DataFrame({
    "StudyHours": [6, 2, 8, 4, 7],
    "Attendance": [85, 40, 92, 60, 75]
})
print("nwe students data")
print(new_students)

#prediction result
preiction=model.predict(new_students)

#diplay prediction clearly
print("prediction result:")
for i in range(len(preiction)):
    if preiction[i]==1:
        print("student",i+1,"will pass")
    else:
        print("student",i+1," will fail")   

################################################################################################
# Question 5 :Calculate- Training Accuracy & Testing Accuracy
################################################################################################
print(Border)
print("Step 5 : Calculate- Training Accuracy & Testing Accuracy")
print(Border)
Y_pred=model.predict(X_test)
correct = sum(Y_test == Y_pred)
total = len(Y_test)

accuracy = correct / total

print(" Manual Accuracy of model is :", accuracy * 100)

#compaire the accuracy with manual accuracy

sklearn_accuracy = accuracy_score(Y_test, Y_pred)

print("Sklearn Accuracy is :", sklearn_accuracy * 100)


################################################################################################
#Question 6 : Identify the students where y_test!=y_pred
################################################################################################
print(Border)
print("Step 6 : Identify the students where y_test!=y_pred")
print(Border)

#Y_test != Y_pred
W=Y_test != Y_pred
#display missclasify rows of students
missclassify_std=X_test[W]
print("missclasify students")
print(missclassify_std)

misclassified_students = missclassify_std.copy()
misclassified_students["Actual"] = Y_test[W]
misclassified_students["Predicted"] = Y_pred[W]

print("\nMisclassified Students with Results:")
print(misclassified_students)

count_wrong = sum(W)

print("Number of Misclassified Students:", count_wrong)
################################################################################################
# Question 7 : Compare accuracy using different random_state
################################################################################################
print(Border)
print(" Step 7 : Compare accuracy using different random_state ")
print(Border)
states=[0, 10, 42]

for rs in states:

    #split dataset
    X_train,X_test,Y_train,Y_test = train_test_split(
        X,Y, test_size=0.2, random_state=rs
    )

    #creste model
    model=DecisionTreeClassifier(random_state=rs)

    #train model
    model.fit(X_train,Y_train)

    #predict 
    Y_pred=model.predict(X_test)

    #accuracy
    acc=accuracy_score(Y_test,Y_pred)

    print(f"Random State={rs} -> Testing Accuracy = {acc*100:.2f}%")


################################################################################################
# Question 8 : Decision tree Visualization, Use: from sklearn.tree import plot_tree
################################################################################################
print(Border)
print("Step 8 :  Decision tree Visualization, Use: from sklearn.tree import plot_tree")
print(Border)
plt.figure(figsize=(6,7))

plot_tree(model,
        feature_names=X.columns,
        class_names=["Fail", "Pass"],
        filled=True,
        rounded=True,
        fontsize=10
        )
plt.title("Decision Tree for Student Performance")
plt.show()
################################################################################################
# Question 9:  Create new column PerformanceIndex
################################################################################################
print(Border)
print("Step 9 :   Create new column PerformanceIndex")
print(Border)

df["PerformanceIndex"]=(df["StudyHours"]*2) + df["Attendance"]

feature_cols=["StudyHours", "Attendance", "PreviousScore", "AssignmentsCompleted", "SleepHours", "PerformanceIndex"]

x=df[feature_cols]
y=df["FinalResult"]

model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)
print(df.head())


################################################################################################
# Step 10: Train model with : max_depth=None
################################################################################################
print(Border)
print("Step 10 : Train model with : max_depth=None")
print(Border)

X=df.drop(columns=["FinalResult"])
Y=df["FinalResult"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=None)
model.fit(X_train, Y_train)

print("Training Accuracy :", model.score(X_train, Y_train)*100)
print("Testing Accuracy :", model.score(X_test, Y_test)*100)
