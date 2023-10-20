# Import of primary libraries for data manipulation
import pandas as pd
import numpy as np
import pyfiglet
# Preprocessing
from sklearn.preprocessing import LabelEncoder
# Produces of the logistc regression model
from sklearn.linear_model import LogisticRegression

# Reading and import the file
csv_file = pd.read_csv('./IrisFlower_ML/set/IRIS.csv')
df = csv_file

# Sum all the columns
#df['soma'] = df.sum(axis=1)
df['soma'] = df['sepal_length'] + df['sepal_width'] + df['petal_length'] + df['petal_width'] 

# Extraction of sl, sw, pl, pw e s
so, s = df[['soma']].values, df[['species']].values

# Convert "s" to numeric values
le = LabelEncoder()
s = le.fit_transform(s.ravel())

# Classifier
clf = LogisticRegression()
clf.fit(so, s)

ascii_banner = pyfiglet.figlet_format("IRIS FLOWER MACHINE LEARNING MODEL")
print(ascii_banner)

# Func of the iris classification.
def class_iris():   
    
    ask = True
    while ask:
        
        # Data input (values)
        sl = float(input("Set the sepal length value (sepal_length): "))
        sw = float(input("Set the sepal width value (sepal_width): "))
        pl = float(input("Set the petal length value (petal_length): "))
        pw = float(input("Set the petal width value (petal_width): "))
        
        # Convert the values in only one value
        temp = sl + sw + pl + pw
        
        # Transform in a numpy array
        temp = np.array(float(temp)).reshape(-1, 1)
              
        # Perform the classification
        class_temp = clf.predict(temp)
        
        # Inverse transformation to retunr the original string
        class_temp = le.inverse_transform(class_temp)
        
        # Presents classification 
        print(f"The classification of this Iris-flower {temp.ravel()[0]} is: {class_temp[0]}\n")
        
        # ask if you want to continue or exit of this aplication 
        ask = input("You want to made a new classification (y/n): ") == 'y'    
        
# Call the function
class_iris()