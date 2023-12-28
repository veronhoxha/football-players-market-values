
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


def detect_retired(players_df):
    '''
    Function to detect players considered retired under these assumptions/conditions:
     - Last season < 2021: The player hasn't played for more than 6 months
     - No current club (NAN): The player doesn't have a current club
     - No Market value in EUR (NAN): The player doesn't have an estimated market value 
     - No Contract Expiration date (NAN): The player doesn't hold a contract with a club

    Takes as an argument a valid players dataframe with the previously mentioned attributes and returns a copy of the dataframe with a "retired"
    column containing a boolean flag. 
    '''
    players_df["retired"] = False
    players_df.loc[(players_df.last_season < 2021) & (players_df.current_club_name.isna()) & 
                   (players_df.market_value_in_eur.isna()) & (players_df.contract_expiration_date.isna()), "retired"] = True
    
    return players_df


def write_latex_table(table, dtypes: dict=None, decimals: int=3, filepath: str="life_table.txt"):

    if dtypes == None:
        dtypes = {}

    with open(filepath, "w") as outfile:

        for ix, row in table.iterrows():

            rw = []

            for col in table.columns:
                
                if col in dtypes:

                    if dtypes[col] == int:
                        x = f"{row[col]:.0f}"
                    elif dtypes[col] == float:
                        x = f"{row[col]:.{decimals}f}"
                    else:
                        x = f"{row[col]}"            

                else:
                    x = f"{row[col]}"

                rw.append(x)
        
            outfile.write(" & ".join(rw))
            outfile.write(" \\\\")
            outfile.write("\n")


def get_retire_table(players_df, ci: tuple=(0.025, 0.975), bootstrap_settings: dict={"simulations": 1000}):
    '''
    Function that takes a players dataframe with at least an age (int) and retired (boolean) fields. Allows for bootstrap simulations to achieve the 
    desired Confidence Intervals for the Survival rate.
    '''
    
    # Sanity check
    if "age" not in players_df.columns or "retired" not in players_df.columns:
        raise Exception("The dataframe must contain at least a field named 'age' (integer) with the age of the observation, and 'retired' (boolean)\
                        denoting the event of interest.")

    if "sample_size" not in bootstrap_settings:
        bootstrap_settings["sample_size"] = len(players_df)

    def __compute_retire_table(players_df):
    
        # Get the count and amount of retired players by age 
        retire_table = players_df.groupby("age").agg({"age": "count", "retired": "sum"}).rename(columns={"age": "n"})

        # Compute h_hat
        retire_table["h"] = retire_table.retired / retire_table.n
        retire_table = retire_table.reset_index(drop=False)

        # Start with prob. 1
        A = [1]

        # Compute S
        for ix, row in retire_table.iterrows():
            # Row zero is prob. 1
            if ix == 0:
                pass
            else:
                # The Survival probability of this year, is the prob. of having survived last year minus the hazard rate times the prob.
                A.append(A[ix-1] - (row.h * A[ix-1]))
            

        retire_table["A"] = A
        
        return retire_table
    
    # Collection of survival functions for simulations
    survival_sims = []

    # For each simulation, draw a bootstrap sample of players, calculate the retire_table and extract the Survival rate by age
    for _ in range(bootstrap_settings["simulations"]):
        bootstrap_sample = players_df.sample(bootstrap_settings["sample_size"], replace=True, random_state=None)  # Sampling with replacement.
                                                            # We need to set random state to None to avoid resampling the same observations everytime
        
        # Compute the retire table of the sample
        ret_table_sim = __compute_retire_table(bootstrap_sample)
        # Add the resulting Survival rate to the list
        survival_sims.append(ret_table_sim[["age", "A"]])

    # Join the simulations by age to calculate the conf. interval
    survival_conf_int = pd.DataFrame([x for x in range(min(players_df.age), max(players_df.age)+1)], columns=["age"])

    # Join all results
    for ss in survival_sims:
        survival_conf_int = survival_conf_int.merge(ss, on="age", how="left")

    # Concat the table with all observations and the confidence intervals from the bootstrapping
    retire_table = pd.concat([__compute_retire_table(players_df), survival_conf_int.iloc[:,1:].quantile(q=[ci[0], ci[1]], axis=1).T], axis=1)

    # Add expectancy
    expectancy = retire_table['A'][::-1]
    retire_table["e"] = expectancy.cumsum()[::-1]

    return retire_table


def log_rank(pop_A, pop_B):

    pop_A.age.
