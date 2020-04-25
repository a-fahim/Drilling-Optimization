# %%
import os,sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import linear_model

def transform(sample):
      [D,h,Fj,Wd,N,rc,gp,ROP]=sample
      Wdt = 0.25 # the min of Wd is 0.97 and for this number log(0) limits to -inf
      # x2 = (1-D)/10000 
      x2 = 10000-D
      x3 = (D**0.69) * ( gp-9 )
      x4 = D * ( gp-rc )/10000 # x4 = D * ( gp-rc )
      x5 = np.log( (Wd-Wdt)/(4-Wdt) )
      x6 = np.log( N/100 )
      x7 = -h
      x8 = np.log( Fj ) # x8 = log( Fj/1000 )
      return np.array([x2,x3,x4,x5,x6,x7,x8,np.log(ROP)])

csv_file = os.path.join(sys.path[0],'bourgoyne.csv')
df = pd.read_csv(csv_file) # df[N,M]
dataset = transform(np.array(df).T)
X = dataset[0:7].T
y = dataset[7].T

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
reg.fit(X_train,y_train)

print(reg.score(X_test,y_test))

# cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
# scores = cross_val_score(reg, X, y, cv=cv,scoring='r2')
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# print(reg.coef_,reg.intercept_)
# print(reg.predict(X_test),y_test)

# %%
