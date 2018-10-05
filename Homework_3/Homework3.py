import json
import warnings
warnings.filterwarnings('ignore')
fj = open('cities.json')
data = json.load(fj)
eco_cities={
    "clean":[] , #no oil, no mining
    "semi_clean":[] , #1 oil or mining
    "dirty" :[], # "more than 1 oil or mining"
}
industrial_cities={
    "many_companies":[], # more than 4 companie
    "mid_companies": [], #from 2 to 5 companies in a city
    "small_companies":[], #"less then 2"
}
pub_cities={
    "good":[], #contains a lot of different chains
    "normal": [], #contains at least 2 chains
    "bad": [] , #others
}

##################
######## 1 #######
##################
all_cities_dict = dict()
for city in data:
    i = 0
    for company in city['company']:
        if company['industry'] == 'Gas/Oil':
            i +=1
        if company['industry'] == 'Mining':
            i +=1
    all_cities_dict[city['city']] = i
eco_cities['clean'] = list(dict((k, v) for k, v in all_cities_dict.items() if v == 0).keys())
eco_cities['semi_clean'] = list(dict((k, v) for k, v in all_cities_dict.items() if v == 1).keys())
eco_cities['dirty'] = list(dict((k, v) for k, v in all_cities_dict.items() if v > 1).keys())

##################
######## 2 #######
##################
all_companies_dict = dict()
for city in data:
    for company in city['company']:
        all_companies_dict[city['city']] = len(city['company'])
industrial_cities['many_companies']  = list(dict((k, v) for k, v in all_companies_dict.items() if v > 4).keys())
industrial_cities['mid_companies']   = list(dict((k, v) for k, v in all_companies_dict.items() if v >= 2 and v < 4).keys())
industrial_cities['small_companies'] = list(dict((k, v) for k, v in all_companies_dict.items() if v < 2).keys())

##################
######## 3 #######
##################
pub_cities={
    "good":[], #contains a lot of different chains
    "normal": [], #contains at least 2 chains
    "bad": [] , #others
}

all_pubs_dict = {'City_Name' : '',
                'bars' : [],}

# for city in data:
#     i = 0
#     for bar in city['bars']:
#         i += 1
# #         print(bar['chain'])
#     all_pubs_dict[city['city']] = i
all_pubs_dict = dict()
for city in data:
    for company in city['company']:
        all_pubs_dict[city['city']] = len(city['bars'])
print(all_pubs_dict)
pub_cities['good']  = list(dict((k, v) for k, v in all_companies_dict.items() if v > 4).keys())
pub_cities['mid_companies']   = list(dict((k, v) for k, v in all_companies_dict.items() if v >= 2 and v < 4).keys())
pub_cities['small_companies'] = list(dict((k, v) for k, v in all_companies_dict.items() if v < 2).keys())