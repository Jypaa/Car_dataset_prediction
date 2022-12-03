import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

df = pd.read_csv(r'cars_dataset.csv', encoding='latin-1')

    # 0model
    # 1year
    # 2price
    # 3transmission
    # 4mileage
    # 5fuelType
    # 6tax
    # 7mpg
    # 8engineSize
    # 9make
    
orgDf = df
# Vaihdetaan koska other arvoja vaihteisto sarakkeessa on 4 joka ei riitä mallin opettamiseen
df['transmission'] = df['transmission'].replace(['Other'], ['Automatic']) 

#Poistetaan rivit joilla samalaisia malleja on yhteensä alle 10 kpl
indexi = df['model'].value_counts()
poistettavat = indexi[indexi <= 10].index
df = df[~df.model.isin(poistettavat)]

#Vaihdetaan autojen vuodet(19kpl) jotka ovat 2001 tai aikasemmin olemaan vuosimallia 2001,
#tosi elämää ajatellen tuossa vaiheessa muulla kuin vuosimallilla on väliä.
#esim km, merkki tai malli
df['year'] = df['year'].replace([2000,1999,1998,1997,1996], 2001) 

#sähköautoja on vain 5, joten sisällytetään ne "muuhun" luokkaan
df['fuelType'] = df['fuelType'].replace(['Electric'], ['Other']) 


#yllä olevien muutosten jälkeen datasta on poistunut 94 riviä ja data on yksinkertaisempaa, 
#jolloin malli on helpompi opettaa
X = df.iloc[:, [0,1,3,4,5,6,7,8,9]]

#Tarkistetaan ettei onko null arvoja eksynyt dataan
print (f'Onko null arvoja: {X.isnull().sum()}\n')
X = X.fillna(0)

#otetaan hinta dataframsesta talteen
y = df.iloc[:, [2]]

#Ennen transformaatiota otetaan "valmis" dataframen talteen ilman y(price) arvoa
Xorg = X

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'),
                                      ['model','transmission','fuelType','Make'])],
                                       remainder='passthrough')
X = ct.fit_transform(X)

X = X.toarray()

# Testi ja harjoitus jako suhteella 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Luodaan skaalaimet x- ja y-komponenteille
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

#Tasot ja opetus
model = Sequential()

model.add(Dense(1000, input_dim=X.shape[1], activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

history=model.fit(X_train, y_train, epochs=200, batch_size=1000, validation_data=(X_test,y_test))

"""
ann:
    r2: 0.9663398307662627
    mae: 1098.2878402909278
    rmse: 1666.252038240165
"""

# Visualisoidaan mallin oppiminen
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')

plt.legend()
plt.ylim(bottom=0, top=5 * min(history.history['val_loss']))
plt.show()

#Testaus
y_pred = scaler_y.inverse_transform(model.predict(X_test))

#tulosten laskenta
y_test = scaler_y.inverse_transform(y_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

#Tulsoten printtaus
print ('\nann:')
print (f'r2: {r2}')
print (f'mae: {mae}')
print (f'rmse: {rmse}\n')

# tallentaan malli, encooderi, transformeri ja skaalaimet levylle
model.save('Auton_ennustus.h5')

with open('Auton_ennustus.pickle', 'wb') as f:
    pickle.dump(model, f)

with open('Auton_ennustus-ann-ct.pickle', 'wb') as f:
    pickle.dump(ct, f)
    
with open('Auton_ennustus-ann-scaler_x.pickle', 'wb') as f:
    pickle.dump(scaler_x, f)
    
with open('Auton_ennustus-ann-scaler_y.pickle', 'wb') as f:
    pickle.dump(scaler_y, f)