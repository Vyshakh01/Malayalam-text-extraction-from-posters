import pandas as pd
import requests
import os

def download_images(data):
    if not os.path.exists('../data/images'):
        os.makedirs('../data/images')

    for index, row in data.iterrows():
        img_url = row['poster_url']
        img_data = requests.get(img_url).content
        with open(f'../data/images/{row["id"]}.png', 'wb') as handler:
            handler.write(img_data)

if __name__ == "__main__":
    file_path = '../data/data.csv'  # Update this path
    data = pd.read_csv(file_path)
    download_images(data)
