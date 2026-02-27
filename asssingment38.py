import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


Border="-"*40
################################################################################################
# Question 1 : Write a Python program to load the final student_performance_ml.csv using pandas.
################################################################################################
print(Border)
print("Quetsion 1 : A Python program to load the final student_performance_ml.csv using pandas.")
print(Border)

Datasetpath="student_performance_ml.csv"

df=pd.read_csv(Datasetpath)

print(df.head())
print(df.tail())
print("The Total Number of Rows & Columns:",df.shape)
print("List of Column Names:",df.columns)
print("Data Types of Each coumn:",df.dtypes)


################################################################################################
# Question2 : Display total number of students in dataset, Count how many stunedts passed and failed.
################################################################################################
print(Border)
print("Step 2 : Total number of students in dataset, Count how many stunedts passed and failed.")
print(Border)

Total_Students=len(df)
print("Total number of students is :",Total_Students)

passed_students = (df['FinalResult'] == 1).sum()
print("Pass Stuents are:",passed_students)

failed_students = (df['FinalResult'] == 0).sum()
print("Failed Stuents are:",failed_students)


################################################################################################
# Question 3 : Using pandas function, calculate and display: Average StudyHours, Average Attendence, Maximum PreviousScore, Minimum SleepHours
################################################################################################
print(Border)
print("Step 3 : Average StudyHours, Average Attendence, Maximum PreviousScore, Minimum SleepHours")
print(Border)

avrege_StudyHours=df['StudyHours'].mean
print("o Avrege StudyHours of a students is:",avrege_StudyHours)

avrege_attendence=df['Attendance'].mean
print("Avrege attendence of a students is:",avrege_attendence)

max_privius_score=df['PreviousScore'].max
print("Maxim previous score is :",max_privius_score)

min_sleepHours=df['SleepHours'].min
print(" minimum SleepHours of a students is:",min_sleepHours)

################################################################################################
# Question 4 : Use value_count() to analyse the distribution of FinalResult.
# Calculate the percentage of Pass and fail students. Is the dataset balanced ? 
################################################################################################
print(Border)
print("Step 4 : Distribution of  FinalResult")
print(Border)

df=pd.read_csv(Datasetpath)
#count result
resultcount = df['FinalResult'].value_counts
print(resultcount)

#count pass and fail stuents percentage
percentages = df['FinalResult'].value_counts(normalize=True) * 100
print(percentages)

################################################################################################
# Question 5 : Analyse whether : Highter StydyHours increase the chance of passing.
# Higher Attendence improves FinalResult.
################################################################################################
print(Border)
print("Step 5 : Effect of StudyHours and Attendance on FinalResult")
print(Border)

print("\nAverage StudyHours:")
print(df.groupby("FinalResult")["StudyHours"].mean())

# Average Attendance based on result


print("\nAverage Attendance:")
print(df.groupby("FinalResult")["Attendance"].mean())

################################################################################################
# Question 6 : Histogram of StudyHours
################################################################################################
print(Border)
print("Step 6 : Histogram of StudyHours")
print(Border)

sns.histplot(df["StudyHours"])
plt.title("Study Hours")
plt.show()

################################################################################################
# Question 7 : Scatterplot of StudyHours VS PreviousScore
################################################################################################
print(Border)
print(" Step 7 : Scatterplot of StudyHours VS PreviousScore")
print(Border)

sns.scatterplot(x="StudyHours", y="PreviousScore", hue='FinalResult',data=df)

plt.title("Study Hours vs Previous Score")
plt.show()

################################################################################################
#Question 8 : Boxplot for Attendence
################################################################################################
print(Border)
print("Step 8 : Boxplot for Attendence")
print(Border)

sns.boxenplot(df['Attendance'])
plt.show()

################################################################################################
# Question 9 : Relationship between AssignmentCompleted and Final Result
################################################################################################
print(Border)
print("Step 9 : Relationship between AssignmentCompleted and Final Result")
print(Border)

sns.boxenplot(x='FinalResult',y='AssignmentsCompleted',data=df)
plt.title("Finalresult vs AssignmentsCompleted ")
plt.show()

################################################################################################
# Question 10 : SleepHour Vs FinalResult
################################################################################################
print(Border)
print("Step 10 : SleepHour Vs FinalResult")
print(Border)

sns.barplot(x='FinalResult',y='SleepHours',data=df)
plt.title("SleepHour VS FinalResult")
plt.show()