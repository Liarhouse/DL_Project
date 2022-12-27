import pandas as pd


pet_info = pd.read_excel('pet.xlsx')
    #rows = pet_info.columns ==
pet = pet_info[pet_info['Breed'] == 'ibizan_hound']
#print(list['Breed'])
    
Breed = pet.iloc[0][1]
print(Breed)