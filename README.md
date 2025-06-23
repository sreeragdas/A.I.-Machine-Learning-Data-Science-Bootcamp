

Steps in Machine learning projects

	1) Data Collection
	2) Data Modelling
		a. Problem Definition
		b. Data
		c. Evaluation
			Determine how you’ll assess model performance. Choose metrics like accuracy, precision, recall, F1-score, RMSE, AUC, etc., depending on the task.
				🔹 Example: Use accuracy and confusion matrix to evaluate a chest pain classifier.
			
		d. Feature:
		
		e. Modelling
		f. Experiments
	3) Deployment

Machine Learning Types
	1) Supervised:
	-Data with label
	(a) Classification:
		(i) Binary Classification
			Example : if a patient have heart disease or not as this only have 2 class
		(ii) Multi-class Classification
			Example : predicting dolls from a bunch of images 
	(b) Regression:
			Problems in which we try to predict a number or continues number , which means a number which go up or down
		a. Example : while buying a house the price can go up and down , based on the rooms and other factors
	2) Unsupervised
	- Data without label
	- Data which have images , audio etc
	Example : recommendation system
	3) Transfer Learning: 
	4) Reinforcement
	Example : Playing chess
	
	
	
Feature in Data 
Suppose for predicting whether a patient have heart disease or not then we required certain information of data such as

 There are 2 types of Features
Numerical Features : eg (Weight)
Character Features : eg (Sex)

Feature Coverage:
Feature Coverage=Total number of recordsNumber of non-missing values​×100% 

🧾 Example:
ID	Weight	Sex	Heart Rate	Chest Pain
1	72	Male	78	Yes
2	60	Female	85	No
3	—	Male	90	Yes

	• Weight has 2 non-missing out of 3 → coverage = 23×100=66.7%\frac{2}{3} \times 100 = 66.7\%32​×100=66.7%
	• Sex has 3 non-missing → coverage = 100%
	• Heart Rate → 100%
	• Chest Pain → 100%


Modelling:
 Three part of Modelling.
	a) Choosing and training a model
		a. Training Set
	b) Tuning a model
		a. Validation Set
	c) Model comparison
		a. Test set
	
Spiling data in Machine learning:
In Machine Learning, splitting the data is a crucial step to ensure that models are trained effectively and evaluated fairly. Typically, the dataset is split into three main parts:

🔄 1. Training Set
	• Purpose: Used to train the model.
	• Size: Typically 60–80% of the total data.
	• Details: The model learns patterns from this data by adjusting its internal parameters.

🧪 2. Validation Set
	• Purpose: Used to fine-tune model parameters and select the best model.
	• Size: Typically 10–20% of the total data.
	• Details: Helps prevent overfitting by providing feedback on how the model performs on unseen data during training.

✅ 3. Test Set
	• Purpose: Used to evaluate the final model's performance after all tuning.
	• Size: Typically 10–20% of the total data.
	• Details: It simulates how the model would perform in the real world.

📊 Common Split Ratios:
Training	Validation	Test
70%	15%	15%
80%	10%	10%
60%	20%	20%

Generalization:
The ability for a machine learning model to perform well on data it hasn't seen before.

Why we split the data  in ML?
1. Learns Properly (Training Set)
	• The model needs to learn patterns from data — the training set is used for this purpose.
	• But if you only test the model on the data it was trained on, it will memorize the data (overfitting) and fail on unseen data.

🧪 2. Tunes Parameters Without Bias (Validation Set)
	• The validation set helps us to:
		○ Choose the best model (e.g., Random Forest vs. XGBoost).
		○ Tune hyperparameters (like learning rate, number of layers, etc.).
	• Since it is not used during training, it acts like "new data" and checks how well the model generalizes.

🎯 3. Tests Final Performance (Test Set)
	• The test set is used only once — at the end — to simulate how the model will perform in the real world.
	• This gives us an unbiased estimate of the model’s accuracy or performance.


Training	Validation	Test
70%	15%	15%
		
Overfitting , Underfitting and Correct Testing model

Normal (correct)
Data Set	Performance
Training	98%
Testing	96%

Overfitting  

	Data Set	Performance
	Training	93%
	Testing	99%
	
Underfitting
Data Set	Performance
Training	64%
Testing	47%



Here Goodfitting or Balanced one is also called as Goldilocks zone.



How Overfitting and underfitting happens
	a) When training data leaks to test data(overfitting)
	b) Data mismatch (underfitting)

How we can overcome Overfitting and underfitting
-underfitting
	a) Try a more advance mode
	b) Increase model hyperparameter
	c) Reduce amount of feature
	d) Train longer
-Overfitting 
	a)     Collect more data
	b)      Try a less advanced mode


Installation software
For Ml & AI we use Anaconda  (jupyter notebook)
Package required 
	(i) Panda
	(ii) Numpy
	(iii) matplotlib
	(iv) sckit-learn
Open anaconda promt run the command 
	Conda create --prefix ./env numpy panda matplotlib sckit-learn
To activate the project 
	Conda activate "filepath/env"
	Example conda activate "conda activate "C:\Users\hsree\Documents\AILearning\env"
We have to install jupyter nodebook inside this filepath 
	Conda install jupyter notebook

Note : do not create project inside onedrive.

Intro To Pandas
	Different datatypes of pandas
		a)series -1 d 
		b)dataFrame- 2 d
		
Anatomy of a dataFrame


	Column (axis=1)
	Row(axis=0)

When we convert a file to csv ,and the pandas will automatically add one more index to file to avoid that we have to provide index=False.

	
	
		
Pandas Attributes and Functions
Data.dtype is an attribute as it does not have bracket in the end
Data.to_csv() is a function as its end with bracket

Note : data.describe works with numerical columns

Note : iloc refer to the position
Loc refer to the index

crosstab in pandas is a function used to count the frequency of combinations of values between two or more columns in a table.
import pandas as pd

data = {
	'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
	'Preference': ['A', 'B', 'A', 'A', 'B']
}

df = pd.DataFrame(data)

# Crosstab
result = pd.crosstab(df['Gender'], df['Preference'])
print(result)
Output:

css
Copy
Edit
Preference  A  B
Gender          
Female      1  2
Male        2  0

What is groupby?
	groupby is used to split data into groups based on the values in one or more columns, and then apply a function (like sum, mean, count, etc.) to each group.

🔹 Think of it like:
	“Group my data by category, and do something to each group.”

📊 Example:

python
CopyEdit
import pandas as pd
data = {
	'Department': ['Sales', 'Sales', 'HR', 'HR', 'IT'],
	'Salary': [50000, 60000, 45000, 47000, 70000]
}
df = pd.DataFrame(data)
# Group by Department and calculate average salary
result = df.groupby('Department')['Salary'].mean()
print(result)
Output:

yaml
CopyEdit
Department
HR       46000.0
IT       70000.0
Sales    55000.0
Name: Salary, dtype: float64


✅ 1. Attributes in pandas
Attributes describe a DataFrame/Series — they don’t need parentheses ().
Attribute	Description
df.shape	Returns (rows, columns)
df.columns	Lists all column names
df.index	Lists row index
df.dtypes	Shows data types of each column
df.size	Total number of elements
df.ndim	Number of dimensions (1D, 2D)
df.values	Numpy array of the data
df.head()	First 5 rows (function, but commonly used)
df.tail()	Last 5 rows (also a function)


🛠️ 2. Functions/Methods in pandas
Functions perform operations on data and need parentheses ().
Function	Description
df.head(n)	Returns top n rows
df.tail(n)	Returns bottom n rows
df.info()	Summary of DataFrame
df.describe()	Statistical summary (mean, std, min, etc.)
df.groupby('col')	Group data by column
df.sort_values(by='col')	Sort by column
df.isnull() / df.notnull()	Detect missing values
df.fillna(value)	Fill missing values
df.dropna()	Drop missing values
df.drop(columns=['col'])	Drop a column
df.rename(columns={})	Rename columns
df.apply(func)	Apply function to each row/column
pd.concat([df1, df2])	Concatenate DataFrames
pd.merge(df1, df2, on='col')	Merge DataFrames
df['col'].value_counts()	Count unique values in a column
df['col'].unique()	Get unique values in a column
df['col'].nunique()	Number of unique values





Intro To Numpy
a1=np.array([1,2,3])
1 d array 
Shape(1,3)
Name : array , vector

a2=np.array([[1,2,3],[4,5,6]]) 
More than 1 d
Name : array , matrix
Shape(2,3)

a3=np.array([[[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18]]])

More than 1 d
Name : array , matrix
Shape(2,3,3)

Type and dtype difference 
Concept	Purpose	Example	Output
type()	Python object type	type(np.array([1, 2]))	<class 'numpy.ndarray'>
.dtype	Data type of elements in array	np.array([1, 2]).dtype	Int64

Note: when we use random and run the same different type we will get different value each time so to get the constant value we use 
# Set random seed to 0
np.random.seed(0)


# Make 'random' numbers
np.random.randint(10, size=(5, 3))


Sometimes, training on large datasets requires a lot of money and GPU resources. To reduce this cost, some columns from the data are removed. This process is known as dimensionality reduction or column removal


**n_estimators=100 means:**
	• The model will build 100 decision trees during training.
	• The final prediction is based on the majority vote from these 100 trees (for classification tasks).

 **Why use cross_val_score?**
When you train a model, you want to make sure it works well on new, unseen data — not just the data it saw during training.

But if you only split your data once (like 80% train, 20% test), your model’s score might depend too much on that specific split.

 cross_val_score helps by:
Splitting your data into multiple parts (folds)

Training on different parts and testing on the rest, multiple times

Giving you multiple accuracy scores, one from each round

This way, you can:

Check how your model performs across different data splits

Get a more reliable and fair evaluation of your model 
Example : cv=5

*****************ROC example :*********
We want to create a machine learning model to predict if someone has a disease.

We tested it on 100 people.

Here's the truth:
40 people actually have the disease

60 people do not have the disease

🧠 Our model's predictions:
The model says “Yes” (has disease) or “No” (doesn't have disease).

Let’s fill in the results like a table:

|                       | Model says “Yes” | Model says “No” |
| --------------------- | ---------------- | --------------- |
| Actually has disease  | 30 ✅             | 10 ❌            |
| Actually doesn't have | 15 ❌             | 45 ✅            |


Let’s understand each of the 4 boxes:
✅ 1. True Positive (TP) – 30
These are people who actually have the disease

And the model correctly said "Yes"

Meaning: Model got it right

❌ 2. False Negative (FN) – 10
People actually have the disease

But model said "No"

Model made a mistake → it missed the disease

❌ 3. False Positive (FP) – 15
People who are actually healthy

But model wrongly said "Yes"

Model made a mistake → false alarm

✅ 4. True Negative (TN) – 45
People who are actually healthy

And model correctly said "No"

Model got it right


************************What is a Confusion Matrix?**************************************
A confusion matrix is used to evaluate the performance of a classification model. It shows how many predictions were:

|                    | Predicted **Yes** (1) | Predicted **No** (0) |
| ------------------ | --------------------- | -------------------- |
| **Actual Yes** (1) | True Positive (TP)    | False Negative (FN)  |
| **Actual No** (0)  | False Positive (FP)   | True Negative (TN)   |

