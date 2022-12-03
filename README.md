Tarkoituksena on ennustaa autojen hintoja.
Aloitin työn perehtymällä, minkälaisen datan kanssa olen tekemisissä. Data koostuu 72435 rivistä ja kymmenestä sarakkeesta jotaka ovat: 
1.Model, 2.Year, 3.Price, 4.Transmission, 5. Mileage, 6.FuelType, 7.Tax, 8.MPG, 9.EngineSize ja 10.Make.

Ensimmäisenä tarkastin minkälaisia vaihtoehtoja kategorioissa on ja datan jakautuminen näissä. Huomasin vaihteisto kohdasta että oli vaihtoehto "muut", joka 
sisälsi neljä autoa. Sisällytin nämä neljä autoa "Automatic" kategorian alle, koska neljällä on hyvin hankala opettaa mallia.

    df['transmission'] = df['transmission'].replace(['Other'], ['Automatic']) 

Tämän jälkeen aloitin tutkimaan tarkemmin "model" saraketta ja huomasin että sarakkeessa oli auto malleja jotka esiintyvät koko datasetissä alle 10 kertaa.
Päätin poistaaa nämä koska niitä ei voi sisällyttää muihin malleihin ja saadakseni datasta hiukan yksinkertaisempaa.

    indexi = df['model'].value_counts()
    poistettavat = indexi[indexi <= 10].index
    df = df[~df.model.isin(poistettavat)]

Seuraavaksi oli vuorossa vuosimallit, koska data keskittyy uudempiin autoihin päätin tässäkin hiukan yksikertaistaa, koska datassa on 19 kpl autoja jotka on vanhempia kuin 2002(1996-2001), 
vaihdoin kyseisten autojen vuosiluvut 2001. Tämän tehin edelleen saadakseni datasta yksinkertaisempaa ja koska oikeassa tilanteissa näin vanhojen autojen arvoon vaikuttaa suuremmalla painotuksella
muut asiat kuin auton ikä.

    df['year'] = df['year'].replace([2000,1999,1998,1997,1996], 2001) 

Viimeisenä muokkasin "fuelType" saraketta. Koska sarakkeessa oli neljä vaihtoehtoa ("Petrol","Diesel","Other" ja "Electric") ja vaihoehto "Electric" sisälsi viisi autoa niin siirsin ne "Other" luokkaan.

    df['fuelType'] = df['fuelType'].replace(['Electric'], ['Other']) 

Tämän jälkeen tarkastin vielä ettei mikään sarake sisällä "null" arvoja ja jos sisältää niin vaihdan ne nolliksi(ennestään oli tieto ettei kyseinen data sisällä "nullia" tai "nan" arvoja).

Tässä vaiheessa dataa on poistunut 94 riviä ja data on yksinkertaistunut hieman, joka helpottaa mallin opettamista.

Nämä tehtyäni transformoin datan "String" muuttujat, jotta mallin opetus on mahdollista. Näin tehtyäni jaoin datan harjoitus- ja testidataan suhteella 80/20. Tämän jälkeen oli vuorossa scaalaus ja verkon rakentaminen.

    model = Sequential()

    model.add(Dense(1000, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

history=model.fit(X_train, y_train, epochs=200, batch_size=1000, validation_data=(X_test,y_test))

Mallintamisessa käytin 200 epochia, batch size oli 1000 jolloinka ja datasetin muokkaamisen jälkeen rivejä on 72341. Tämä tarkoittaa että koko harjoituksen jälkeen läpi on mennyt kokonaisuudessaaan reilu 
360 000 batchia. Näillä tiedoilla arvoiksi tuli:

    r2: 0.9663398307662627
    mae: 1098.2878402909278
    rmse: 1666.252038240165

Kaiken tämän jälkeen tallennan vielä mallit mahdollisia uusia käyttöjä varten.




