import argparse
import math
import os
from datetime import datetime, timedelta

from geopy.geocoders import Nominatim
from geopy.distance import distance
import requests
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from tqdm import tqdm

url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
output_dir = "data"
raw_data_path = os.path.join(output_dir, "covid_data.csv")
county_coords_data_path = os.path.join(output_dir, "county_coords.csv")
final_data_path = os.path.join(output_dir, "covid19_dataset.json")
# sigma = 100 seems to give quite good edge weights
sigma = 100


def download_raw_data():
    print('Downloading raw covid19 data...')
    req = requests.get(url, stream=True)
    with open(raw_data_path, 'wb') as outfile:
        chunk_size = 1048576
        for chunk in req.iter_content(chunk_size=chunk_size):
            outfile.write(chunk)
        outfile.close()
    print('Done.')


def prepare_counties_coords():
    print('Preparing coordinates for counties...')
    data_csv = pd.read_csv(raw_data_path)
    counties = data_csv[['county', 'state']].drop_duplicates(keep='first', ignore_index=True)

    longitude, latitude = np.array([]), np.array([])
    geolocator = Nominatim(user_agent='covid_19_data', timeout=3)
    counties_list = list(counties.itertuples(index=False))
    for county in tqdm(counties_list):
        countyName = str(county[0]) + ", " + str(county[1])
        geo = geolocator.geocode(countyName)
        if geo:
            longitude = np.append(longitude, str(geo.longitude))
            latitude = np.append(latitude, str(geo.latitude))
        else:
            if countyName == "New Hanover, North Carolina":
                longitude = np.append(longitude, str(-77.9672919))
                latitude = np.append(latitude, str(34.1630138))
            elif countyName == "Bethel Census Area, Alaska":
                longitude = np.append(longitude, str(-162.9636124))
                latitude = np.append(latitude, str(61.3545136))
            elif countyName == "Valdez-Cordova Census Area, Alaska":
                longitude = np.append(longitude, str(-149.4745618))
                latitude = np.append(latitude, str(61.4024872))
            elif countyName == "Bristol Bay plus Lake and Peninsula, Alaska":
                longitude = np.append(longitude, str(-161.1332367))
                latitude = np.append(latitude, str(58.1958546))
            elif countyName == "Yakutat plus Hoonah-Angoon, Alaska":
                longitude = np.append(longitude, str(-135.92696))
                latitude = np.append(latitude, str(57.758603))
            elif countyName == "Yukon-Koyukuk Census Area, Alaska":
                longitude = np.append(longitude, str(-159.9618241))
                latitude = np.append(latitude, str(64.968391))
            elif countyName == "Yukon-Koyukuk Census Area, Alaska":
                longitude = np.append(longitude, str(-148.4852298))
                latitude = np.append(latitude, str(63.8059416))
            else:
                longitude = np.append(longitude, str(0))
                latitude = np.append(latitude, str(0))

    counties = counties.assign(Lon=pd.Series(longitude).values)
    counties = counties.assign(Lat=pd.Series(latitude).values)

    counties.to_csv(county_coords_data_path)
    print('Done.')


def prepare_final_dataset(nlat: float, slat: float, wlon: float, elon: float):
    print('Preparing final dataset...')
    counties = pd.read_csv(county_coords_data_path, index_col=0)
    counties = counties[(counties["Lat"] <= nlat) & (counties["Lat"] >= slat) & (counties["Lon"] <= elon) & (counties["Lon"] >= wlon)].reset_index(drop=True)
    dataset = dict()
    node_ids = dict()
    edges = []
    edges_weights = []
    print('\tPreparing edges and edge_weights...')
    for index in tqdm(range(len(counties))):
        county = counties.iloc[index]
        node_ids[str(county['county']) + ", " + str(county['state'])] = str(index)
        for i in range(index + 1, len(counties)):
            d = distance((county['Lat'], county['Lon']), (counties.loc[i, "Lat"], counties.loc[i, "Lon"])).km
            weight = math.exp(-(d ** 2 / (sigma ** 2)))
            if weight > 0.5:
                edges.append([index, i])
                edges.append([i, index])
                edges_weights.append(weight)
                edges_weights.append(weight)

    dataset['edges'] = edges
    dataset['edges_weights'] = edges_weights
    dataset['node_ids'] = node_ids

    raw_data_csv = pd.read_csv(raw_data_path)

    FX = []
    start_date = datetime.strptime(raw_data_csv.iloc[0]["date"], "%Y-%m-%d")
    end_date = datetime.strptime(raw_data_csv.iloc[-1]["date"], "%Y-%m-%d")

    delta = end_date - start_date
    counties = list(counties[["county", "state"]].itertuples(index=False))
    print('\tPreparing cases arrays...')
    cases = np.zeros(len(counties))
    for _ in tqdm(range(delta.days)):
        daily_records = raw_data_csv[raw_data_csv['date'] == start_date.strftime("%Y-%m-%d")]
        daily_cases = []
        for index, county in enumerate(counties):
            rows_for_county = daily_records[
                (daily_records["county"] == county[0]) & (daily_records["state"] == county[1])]
            if len(rows_for_county) > 0:
                new_cases = rows_for_county.iloc[0]["cases"] - cases[index]
                cases[index] = cases[index] + new_cases
                daily_cases.append(new_cases)
            else:
                daily_cases.append(0)
        FX.append(daily_cases)
        start_date = start_date + timedelta(days=1)

    mean = np.array(FX).mean()
    std = np.array(FX).std()
    FX = (FX - mean) / std

    dataset['FX'] = FX.tolist()
    dataset['mean'] = mean
    dataset['std'] = std

    with open(final_data_path, 'w') as fp:
        json.dump(dataset, fp)
    print('Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--nlat', help='Optional, use for spatially limiting counties. 90 to -90. Specifies upper bound of '
                                       'spherical rectangle.', default=90., type=float)
    parser.add_argument('--slat', help='Optional, use for spatially limiting counties. 90 to -90. Specifies lower bound of '
                                       'spherical rectangle.', default=-90., type=float)
    parser.add_argument('--wlon', help='Optional, use for spatially limiting counties. -180 to 180. Specifies western '
                                       'bound of spherical rectangle.', default=-180., type=float)
    parser.add_argument('--elon', help='Optional, use for spatially limiting counties -180 to 180. Specifies eastern '
                                       'bound of spherical rectangle.', default=180., type=float)
    parser.add_argument('--plot_name', help='Filename for saving heatmap of adjacency matrix', default='heatmap.png', type=str)
    args = parser.parse_args()

    if not os.path.exists(county_coords_data_path):
        prepare_counties_coords()
    else:
        print('Counties coords already prepared.')

    if not os.path.exists(final_data_path):
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(raw_data_path):
            download_raw_data()
        else:
            print('Covid19 data already downloaded.')

        prepare_final_dataset(args.nlat, args.slat, args.wlon, args.elon)

    print("Preparing adjacency matrix...")
    counties = pd.read_csv(county_coords_data_path, index_col=0)
    counties = counties[(counties["Lat"] <= args.nlat) & (counties["Lat"] >= args.slat) & (counties["Lon"] <= args.elon) & (counties["Lon"] >= args.wlon)].reset_index(drop=True)
    adjacencies = []
    for index in tqdm(range(len(counties))):
        county = counties.iloc[index]
        adjacency_row = []
        for i in range(len(counties)):
            if i == index:
                adjacency_row.append(1)
            else:
                d = distance((county['Lat'], county['Lon']), (counties.loc[i, "Lat"], counties.loc[i, "Lon"])).km
                weight = math.exp(-(d ** 2 / (sigma ** 2)))
                if weight > 0.5:
                    adjacency_row.append(weight)
                else:
                    adjacency_row.append(0)
        adjacencies.append(adjacency_row)

    plt.imshow(adjacencies, cmap=cm.Blues, interpolation='nearest')
    plt.ylabel("Numer węzła")
    plt.xlabel("Numer węzła")
    # plt.yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    # plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    plt.savefig('figures/' + args.plot_name)
    plt.show()

