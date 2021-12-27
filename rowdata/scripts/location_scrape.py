###################
##TRYING TO GET RACE INFO ON LOCATION


url='https://www.row2k.com/results/index.cfm?year=2019'
from bs4 import BeautifulSoup
import requests
import re
page=requests.get(url)
soup=BeautifulSoup(page.content,'html.parser')
main_sections=soup.find_all('tr', attrs={'bgcolor': 'ffcc00'})


for row in main_sections:

    # Given a row that is a date this extract the date
    val=row.text
    date=val.split(',')[1:]
    month=date[0].split(' ')[1]
    day=date[0].split(' ')[2]
    year=date[1].replace(' ','')

    section=soup.find(text=val).findNext('tr')

    sub_sections=[]
    for i in range(len(section.find_all('li'))):
        if len(section.find_all('li')[i].find_all('span'))>0:
            sub_sections.append(i)


    for i in sub_sections:
        race_type=section.find_all('li')[i].findNext('span').text
        if 'COLLEGIATE' not in race_type.upper():
            pass

        else:
            ##### NEED TO WORK ON THIS PART ONCE WE GET TO A SPECIFIC TYPE OF RACE SECTION
            ##### I.E COLLEGIATE MENS HOW TO GET THE INDIVIDUAL RACES ON THAT DAY PER TYPES

            races=section.find_all('li')[i].find_all('li')
            text_string=races[0].text
            text_string=re.sub('\t',' ',text_string)
            text_string=re.sub('\r',' ',text_string)
            text_string=text_string.splitlines()
            text_string = list(map(str.lstrip, text_string))
            text_string = list(map(str.rstrip, text_string))
            text_string=text_string[:-1]






#
