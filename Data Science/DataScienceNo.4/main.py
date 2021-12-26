import matplotlib
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
#%matplotlib inline
from scipy.stats import pearsonr

df = pd.read_csv(r"C:\Users\eytan\PycharmProjects\DataScienceNo.4\Bike_sharing_data.csv")



#plt.hist(df['cnt'],bins=11)
#plt.show()


print(df.casual.describe())


#print((df['hum']>0.700).value_counts(normalize=True))
#print((df['cnt']==0).value_counts(normalize=True))
#print((df['weathersit']==1).value_counts(normalize=True))


#plt.scatter(df['temp'],df['windspeed'])
#plt.show()


#Question 1
df = pd.read_csv(r"C:\Users\eytan\PycharmProjects\DataScienceNo.4\DataScienceNo.4.csv")
#print(df.age.value_counts())
print(df['age'].describe())
print((df.mark>90).value_counts(normalize=True))
print((df.age<25).value_counts(normalize=True))


corr =  pearsonr(df.mark,df.age)

print(corr)