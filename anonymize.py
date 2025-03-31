import pandas as pd
import random

#%%
latent_vectors = pd.read_csv('./data/processed/latent_vectors.csv')
rlps_data = pd.read_csv('./data/processed/rlps_2023_data.csv')


boxid_counts_per_gemeente = rlps_data.groupby('GEMEENTE')['BOXID'].nunique()
boxid_unique_counts = len(rlps_data['BOXID'].unique())

planet_names = [
    "Kepler-442b", "Gliese-581g", "Proxima-b", "Tau Ceti-f", "HD-40307g", "LHS-1140b",
    "Wolf-1061c", "Trappist-1d", "Ross-128b", "Kapteyn-b", "K2-18b", "55-Cancri-e",
    "GJ-1214b", "TOI-700d", "CoRoT-7b", "YZ-Ceti-b", "Gliese-667Cc", "HD-219134b", "LP-890-9c"
]

# Base list of known bright stars
base_star_names = [
    "Sirius", "Betelgeuse", "Vega", "Rigel", "Antares", "Altair", "Canopus", "Aldebaran",
    "Arcturus", "Procyon", "Polaris", "Spica", "Capella", "Deneb", "Bellatrix", "Fomalhaut",
    "Alnitak", "Alnilam", "Mintaka", "Castor", "Pollux", "Hadar", "Achernar", "Alphard",
    "Mizar", "Alcor", "Merak", "Dubhe", "Rasalhague", "Markab", "Algol", "Regulus"
]

# Generate variations by appending Greek letters and numbers
greek_letters = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa"]
star_variations = [f"{greek}_{star}" for greek in greek_letters for star in base_star_names]

# Expand further by appending numbers
expanded_star_names = star_variations + [f"{name}_{i}" for i in range(1, 200) for name in base_star_names]

# Ensure we have at least 6000 unique names
random.shuffle(expanded_star_names)
unique_star_names = list(set(expanded_star_names))[:6000]

#%%
# Get unique city names and map them to planets
unique_cities = rlps_data['GEMEENTE'].unique()
city_to_planet_map = {city: planet for city, planet in zip(unique_cities, planet_names)}


unique_boxids = rlps_data['BOXID'].unique()

# Ensure the number of star names matches the number of unique BOXIDs
if len(unique_star_names) < len(unique_boxids):
    raise ValueError("Not enough unique star names to anonymize all BOXIDs. Expand the name list.")

# Shuffle star names for randomness
random.shuffle(unique_star_names)

boxid_to_star_map = {boxid: star for boxid, star in zip(unique_boxids, unique_star_names)}

# Apply the anonymization
rlps_data['ANONYMIZED_BOXID'] = rlps_data['BOXID'].map(boxid_to_star_map)
rlps_data['ANONYMIZED_CITY'] = rlps_data['GEMEENTE'].map(city_to_planet_map)

latent_vectors['ANONYMIZED_BOXID'] = latent_vectors['BOXID'].map(boxid_to_star_map)
latent_vectors['ANONYMIZED_CITY'] = latent_vectors['GEMEENTE'].map(city_to_planet_map)


rlps_data = rlps_data.drop(columns=["GEMEENTE", "BOXID"])
latent_vectors = latent_vectors.drop(columns=["GEMEENTE", "BOXID"])

#%% Save
rlps_data.to_csv('./data/processed/rlps_2023_data_anonymized.csv', index=False)
latent_vectors.to_csv('./data/processed/latent_vectors_anonymized.csv', index=False)



