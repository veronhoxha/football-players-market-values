
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup # xml parsing
from IPython.display import clear_output
import Levenshtein
from datetime import datetime

def extract_website_structure(filepath: str="FIFA_attribute_structure.txt"):

    website_structure_path = filepath
    website_attribute_structure = {}

    file = open(website_structure_path)

    for line in file.readlines():

        line = line.strip()

        if len(line.split()) == 1:
            category = line.capitalize()
            continue
        
        else:
            val, attribute = line.split(" ", maxsplit=1)
            val = int(val)
            attribute = attribute.capitalize()

        if category in website_attribute_structure:
            website_attribute_structure[category].append(attribute)
            #website_attribute_structure[category][attribute] = val
        else:
            website_attribute_structure[category] = [attribute]
            #website_attribute_structure[category] = {attribute: val}

    return website_attribute_structure


class FIFAScraper:

    def __init__(self):
        self.HEADERS = {'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'}
        self.structure = extract_website_structure()

        pass


    def get_player_attributes(self, player_url):

        response = requests.get(player_url, headers=self.HEADERS)
        
        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}")
        
        attributes = {"player_url": player_url}

        sections = BeautifulSoup(response.text, 'html.parser').find_all('div', class_="card")

        for section in sections:

            try:
                category = section.find("h5").get_text()

            except:
                pass
            
            if category in self.structure:

                for li in section.find_all("li"):

                    val, attribute = li.find_all("span")

                    attributes[attribute.get_text()] = val.get_text()

        return attributes
    

    def create_players_df(self, players_df, expand=True):

        if "player_url" not in players_df.columns:
            raise Exception("'player_url' column not found.")
        
        players_attributes = []

        for player in players_df.player_url:

            try:
                players_attributes.append(self.get_player_attributes(player))
            except:
                pass

        return pd.DataFrame(players_attributes)
    

def match_players(transfermarkt, fifa):
    
    # Map of long names to short names or nicknames used
    names_map = {
        "Cam": "Cameron",
        "Cammy": "Cameron",
        "Joe": "Joseph",
        "Danny": "Daniel",
        "Tony": "Anthony",
        "Nicky": "Nicholas",
        "Andy": "Andrew",
        "Alex": "Alejandro",
        "Harry": "Harrison",
        "Kenny": "Kenneth",
        "Willi": "Wilhem",
        "Terry": "Terence",
        "Tom": "Thomas",
        "Jamie": "James",
        "Frank": "Francis",
        "Matty": "Matthew",
        "Mattie": "Matthew",
        "Nacho": "Ignacio",
        "Charlie": "Charles",
        "Paddy": "Patrick",
        "Yevgen": "Yevhen",
        "Jonny": "Jonathan",
        "Nick": "Nicholas",
        "Tino": "Valentino",
        "Freddie": "Frederick",
        "Eddie": "Edward",
        "Solly": "Solomon"
    }

    # Include the inverse as well
    names_map.update({v: k for k,v in names_map.items()})

    # Normalize names with unicode chrs
    transfermarkt["name_"] = transfermarkt["name"].apply(lambda x: x.replace("-", " ").replace("ø", "o").replace("ö", "o")
                                                         .replace('ç', "c").replace('ł', "l").replace("Ł", "L").replace("Ø", "O")
                                                         .replace("ı", "i").title() if type(x) == str else x)
    transfermarkt["first_name_"] = transfermarkt["first_name"].apply(lambda x: x.replace("-", " ").replace("ø", "o")
                                                                     .replace("ö", "o").replace('ç', "c").replace('ł', "l").replace("Ł", "L").replace("Ø", "O")
                                                                     .replace("ı", "i").title() if type(x) == str else x)
    transfermarkt["last_name_"] = transfermarkt["last_name"].apply(lambda x: x.replace("-", " ").replace("ø", "o").replace("ö", "o")
                                                                   .replace('ç', "c").replace('ł', "l").replace("Ł", "L").replace("Ø", "O")
                                                                   .replace("ı", "i").title() if type(x) == str else x)
    fifa["short_name_"] = fifa["short_name"].apply(lambda x: x.replace("-", " ").replace("ø", "o").replace("ö", "o").replace('ç', "c")
                                                   .replace('ł', "l").replace("Ł", "L").replace("Ø", "O")
                                                   .replace("ı", "i").title() if type(x) == str else x)
    fifa["long_name_"] = fifa["long_name"].apply(lambda x: x.replace("-", " ").replace("ø", "o").replace("ö", "o").replace('ç', "c")
                                                 .replace('ł', "l").replace("Ł", "L").replace("Ø", "O")
                                                 .replace("ı", "i").title() if type(x) == str else x)

    transfermarkt["name_"] = transfermarkt["name_"].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').astype(str)
    transfermarkt["first_name_"] = transfermarkt["first_name_"].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').astype(str)
    transfermarkt["last_name_"] = transfermarkt["last_name_"].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').astype(str)
    fifa["short_name_"] = fifa["short_name_"].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    fifa["long_name_"] = fifa["long_name_"].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

    # New column that matches FIFA's full name
    # Container for matches
    longname_matches = []
    revisit = [False for i in range(len(transfermarkt))]

    # For each player, find potential matches
    for ix, row in transfermarkt.iterrows():
        
        print(f"{ix/len(transfermarkt):.2%}\tLooking for {row['name']} ({row['name_']}) / First: {row['first_name_']} / Last: {row['last_name']}...")

        ### Matching hierarchy
        # Must be same date of birth
        subset = fifa[fifa.dob == row.date_of_birth]
        
        # Exact matches: Either long name or short name matches exactly to the row name
        exact = subset[((subset.long_name_ == row["name_"]) | (subset.short_name_ == row["name_"]))]

        # If found a match
        if len(exact) == 1:
            LongName = exact.long_name.values[0]
            print(f"FOUND: {LongName}")

        # If no exact match
        else:
            # Relax name constraint
            print("Not exact match, relaxing constraints...")
            
            # If first name is in names map, try also with the variants
            if row["first_name_"] in names_map:
                contained = subset[(subset.long_name_.str.contains("|".join([row["first_name_"], names_map[row["first_name_"]]])))
                                    & (subset.long_name_.str.contains(row["last_name_"]))]
            else:
                contained = subset[(subset.long_name_.str.contains(row["first_name_"])) & (subset.long_name_.str.contains(row["last_name_"]))]

            # If exact match
            if len(contained) == 1:
                LongName = contained.long_name.values[0]

            # If still no match
            else:
                print("Not relaxed match, fuzzy matching...")

                # Try fuzzy matching
                fullname = row["first_name_"] + row["last_name_"]
                subset["fuzzy"] = subset.long_name_.apply(lambda x: Levenshtein.distance(fullname, x))
                fuzzy = subset[subset.fuzzy <= 4]

                if len(fuzzy) == 1:
                    LongName = fuzzy.long_name.values[0]

                else:
                    print("No fuzzy matching, looking only last names...")
                    # Relax further: Only look for contained last names
                    potential_matches = subset[subset.long_name_.str.contains(row["last_name_"])]

                    if len(potential_matches) > 0:
                        # Flag for revisit
                        revisit[ix] = (row["name"], potential_matches)
                        LongName = "REVISIT"

                    # If still no match, abandon
                    else:
                        print("Nothing found, abandoning...")
                        LongName = np.nan


        longname_matches.append(LongName)
        if ix != len(longname_matches)-1:
            raise Exception("WTF")

        clear_output(wait=True)

    # The revisits (manual inspection)
    for i, rv in enumerate(revisit):

        # If needs inspection
        if rv:
            print(f"{i/len(revisit):.2%}\tLooking for {rv[0]}")
            print("Potential matches: ")
            # Display results
            display(rv[1])

            opt = False

            # Accept input option
            while not opt:
                opt = input("Select option or C to escape: ")

                if opt.lower() == "c":
                    # If not a valid option
                    LongName = np.nan

                else:
                    try:
                        LongName = rv[1].loc[int(opt), "long_name"]
                        revisit[i] = False # Erase from revisit
                    except:
                        opt = False

            longname_matches[i] = LongName

            clear_output(wait=True)


    # Final chance, we just do fuzzy matching for names with same birthdate and nationality
    for i, rv in enumerate(longname_matches):

        # If already found, continue
        if type(rv) == str:
            continue
        else:

            subset = fifa[(fifa.dob == transfermarkt.loc[i, "date_of_birth"]) & (fifa.nationality_name == transfermarkt.loc[i, "country_of_citizenship"])]

            if len(subset) == 0:
                continue

            else:
                for i, r in subset.iterrows():
                    if (Levenshtein.ratio(r.short_name, row.first_name) > 0.5) | (Levenshtein.ratio(r.short_name, row.last_name) > 0.5) | (Levenshtein.ratio(r.short_name, row["name"]) > 0.5) | (Levenshtein.ratio(r.long_name, row.first_name) > 0.5) | (Levenshtein.ratio(r.long_name, row.last_name) > 0.5) | (Levenshtein.ratio(r.long_name, row["name"]) > 0.5):
                        print(Levenshtein.ratio(r.short_name, row.first_name), Levenshtein.ratio(r.short_name, row.last_name), Levenshtein.ratio(r.short_name, row["name"]), Levenshtein.ratio(r.long_name, row.first_name), Levenshtein.ratio(r.long_name, row.last_name), Levenshtein.ratio(r.long_name, row["name"]))
                        print("Potential matches: ")
                        display(subset)

                        opt = False

                        while not opt:
                            opt = input("Select option or C to escape: ")

                            if opt.lower() == "c":
                                LongName = np.nan

                            else:
                                try:
                                    LongName = subset.loc[int(opt), "long_name"]
                                except:
                                    opt = False

                    else:
                        LongName = np.nan

                longname_matches[i] = LongName

            clear_output(wait=True)


    # Cleanup
    transfermarkt.drop(["name_", "first_name_", "last_name_"], axis=1, inplace=True)
    fifa.drop(["short_name_", "long_name_"], axis=1, inplace=True)

    return longname_matches



def calculate_age(date_of_birth, baseline="2021-05-08"): 
    '''
    Calculate the age of a player from his date of birth. By default, it calculates his age to the baseline date of data collection.
    '''
    if type(date_of_birth) != str:
        return np.nan
    
    date_of_birth = datetime.strptime(date_of_birth, "%Y-%m-%d").date() 
    baseline = datetime.strptime(baseline, "%Y-%m-%d").date() 
    return baseline.year - date_of_birth.year - ((baseline.month,  
                                      baseline.day) < (date_of_birth.month,  
                                                    date_of_birth.day))


<<<<<<< HEAD
=======
def hist_dist(df, cols):

    fig, ax = plt.subplots
    for c in cols:
        pass

>>>>>>> 01f49830a31a8d107a991a053abeae8183f3e3f5
