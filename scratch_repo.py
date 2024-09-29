
import re


    
def rear_camera_count(rear_cam_data):
    '''Gets the number of rear camera.'''
    pattern = r'(Quad Camera|Triple Camera|Dual Camera)'
    match = re.search(pattern, str(rear_cam_data), re.IGNORECASE)
    if match:
        matched = match.group(1).lower()
        if matched == 'quad camera':
            return 4
        if matched == 'triple camera':
            return 3
        if matched == 'dual camera':
            return 2
    else:
        return 1

def rear_camera_main_mp(rear_cam_data):
    '''Gets the maximum megapixel of the rear camera.'''
    pattern = r'(\d+(\.\d+)?)\s*(mp|megapixels)'
    match = re.search(pattern, str(rear_cam_data), re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return 0
    
def front_camera_count(front_cam_data):
    '''Gets the count of the front camera.'''
    pattern = r'(Multiple Camera|Dual Camera)'
    match = re.search(pattern, str(front_cam_data), re.IGNORECASE)
    if match:
        matched = match.group(1).lower()
        if matched == 'multiple camera':
            return 3
        if matched == 'dual camera':
            return 2
    else:
        return 1
    
def front_camera_main_mp(front_cam_data):
    '''Gets the megapixel of the camera.'''
    pattern = r'(\d+(\.\d+)?)\s*(mp|megapixels)'
    match = re.search(pattern, str(front_cam_data), re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return 0
    
def batt_mah(batt_data):
    '''Gets the mAh of battery of '''
    pattern = r'(\d+(\.\d+)?)\s*(mah)'
    match = re.search(pattern, batt_data, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return 0

def batt_fast_charging(batt_data):
    '''Gets if phone has fast charging or not capability.'''
    pattern = r'fast ?charging|fastcharging'
    match = re.search(pattern, batt_data, re.IGNORECASE)
    if match:
        return 1
    else:
        return 0
    
def replace_infreq_values(df_col, threshold=10, replacement_value='Others'):
    '''
    If an entry has very little count when compared to a threshold, it
    replaces that entry with Others.
    '''
    value_counts = df_col.value_counts()
    infrequent_category = value_counts[value_counts < threshold].index
    df_col = df_col.apply(
        lambda x: replacement_value if x in infrequent_category else x
    )
    return df_col

def calculate_performance_score(cpu, minimum):
    '''
    Calculates a simple benchmark score for the cpu based on clock speed and 
    number of cores.
    '''
    core_speed_pairs = re.split(r'[,&]', cpu)
    try:
        total_score = 0
        for pair in core_speed_pairs:
            match = re.search(r'(\d*\.?\d*)[ ]?GHz', pair, re.I)
            if match:
                clock_speed = float(match.group(1))
                try:
                    num_cores = int(re.search(r'(\d+)x', pair).group(1))
                except:
                    num_cores = 4
                total_score += num_cores * clock_speed
        if total_score < float(minimum):
            return float(minimum)
        else:
            return total_score
    
    except:
        return float(minimum)
    

#######################################################################
# Data Cleaning
#######################################################################
# Create a deep copy of the scraped data

gpu_score_df= pd.read_csv(
    'gpu_score.csv', 
    header=0, 
)
gpu_manual_score_df = pd.DataFrame(
    list(gpu_score_update.items()),
    columns=['gpu', 'gpu_score']
)
gpu_score_df2 = pd.concat(
    [gpu_score_df, gpu_manual_score_df]
)

def preprocess(df):

    df.fillna(np.nan, inplace=True)

    # Drop columns with few values
    col_to_drop = [
        'Link', 'Infrared', ' Camera', 'TV', 'Networks', 'GPS', 'Buy Online'
    ]

    try:
        df = df.drop(col_to_drop, axis=1)
    except:
        pass

    df['brand'] = (
        df['name']
        .str.split(' ', n=-1)
        .str[0]
        .str.lower()
    )

    # Manual cleaning for Stars
    df['stars'] = (
        df['stars']
        .fillna(df['stars'].median())
        .astype('float64')
    )

    # Manual cleaning for Stars Count
    df['stars_count'] = (
        df['stars_count']
        .fillna(df['stars_count'].median())
        .astype('int16')
    )

    # Manual cleaning for LikeShare
    like_share_mean = (
        df['like_share']
        .str.split()
        .str[0]
        .str.replace(',', '')
        .astype(float)
        .mean()
    )

    df['like_share'] = (
        df['like_share']
        .str.split()
        .str[0]
        .str.replace(',', '')
        .fillna(like_share_mean)
        .astype(int)
    )

    df['price'] = (
        df['name'].map(missing_price_dict)
        .fillna(df['price'])
        .str.replace('₱', '')
        .str.replace('|', '')
        .str.replace('Official', '')
        .str.replace(' -  Price in the Philippines', '')
        .str.replace(',', '')
        .str.replace('.00', '')
        .str.strip()
        .apply(convert_price)
    )

    df['OS'] = (
        df['OS']
        .str.split(' with ')
        .str[0]
    )

    # Manual cleaning for CPU
    df['CPU'] = (
        df['CPU']
        .fillna(
            'Hexa-core (2x high power Lightning cores at 2.66 GHz + 4x low power Thunder cores at 1.82 GHz)'
        )
    )

    # Manual cleaning for GPU
    df['GPU'] = (
        df['GPU']
        .fillna('Unknown')
        .str.strip()
        .str.replace('_', 'Unknown')
    )

    # Manual cleaning for Rear Camera
    df['Rear Camera'] = (
        df['Rear Camera']
        .fillna('Unknown')
    )

    # Manual cleaning for Front Camera
    df['Front Camera'] = (
        df['Front Camera']
        .fillna('Unknown')
    )


    # Manual cleaning for Expansion
    df['Expansion'] = (
        df['Expansion']
        .replace(replacement_expansion)
    )

    # Manual cleaning for SIM Card
    df['SIM Card'] = (
        df['SIM Card']
        .replace(replacement_sim_card)
    )

    # Manual cleaning for Cellular
    df['Cellular'] = (
        df['Cellular']
        .fillna(' ')
        .str.split(',')
        .str[0]
        .replace(replacement_cellular)
    )

    # Manual cleaning for Wi-Fi
    df['Wi-Fi'] = (
        df['Wi-Fi']
        .replace(replacement_wifi)
    )

    # Manual cleaning for NFC
    df['NFC'] = (
        df['NFC']
        .str.replace('Yes', '1')
        .str.replace('No', '0')
        .astype('int16')
        .astype('bool')
    )

    # Manual cleaning for Positioning
    df['Positioning'] = (
        df['Positioning']
        .str.strip()
        .fillna(' ')
        .str.split(',')
    )

    # Manual cleaning for USB OTG
    df['USB OTG'] = (
        df['USB OTG']
        .fillna(False)
        .astype('bool')
    )

    # Manual cleaning for Sound
    df['Sensors'] = (
        df['Sensors']
        .str.split(r'[;,]| and ', regex=True)
    )

    # Manual cleaning for Sound
    df['Sound'] = (
        df['Sound']
        .fillna('Unknown')
    )

    # Manual cleaning for FM Radio
    df['FM Radio'] = (
        df['FM Radio']
        .str.replace('Yes', '1')
        .str.replace('Yes with RDS', '1')
        .str.replace('No', '0')
        .astype('bool')
    )

    df['Biometrics'] = (
        df['Biometrics']
        .str.replace('None', 'Unknown')
        .replace(replace_biometrics)
        .str.strip()
    )

    # Manual cleaning for Material
    df['Material'] = (
        df['Material']
        .fillna('Unknown')
    )

    # Manual cleaning for Dimensions
    df['Dimensions'] = (
        df['Dimensions']
        .str.replace('_', '0 0 0')
        .fillna('0 0 0')
        .str.replace(r'[^0-9.]+', ' ', regex=True)
        .str.strip()
        .str.split(' ')
        .str[:3]
    )

    # Manual cleaning for Weight 
    df['Weight'] = (
        df['Weight']
        .str.replace(r'[^0-9.]+', ' ', regex=True)
        .str.strip()
        .str.split()
        .str[0]
    )

    df['Weight'] = (
        df['Weight']
        .fillna(pd.to_numeric(df['Weight']).mean())
        .astype('float16')
    )

    # Manual cleaning for Launch Date

    df['Launch Date'] = (
        df['Launch Date']
        .ffill()
        .replace(replace_launchdate)
    )

    df['Launch Date'] = pd.to_datetime(
        df['Launch Date'],
        format='mixed'
    )

    # Manual cleaning for Price
            
    df['Price'] = (
        df['Price']
        .str.replace(',', '')
        .str.split('-')
        .str[0]
        .str.strip(' ')
        .str.replace('₱', '')
        .str.replace('No official price in the Philippines yet.', '', regex=False)
        .str.replace('/', '')
        .str.strip()
        .str.split(' ')
        .str[:2]
        .apply(convert_price)
    )

    df['Price'] = (
        df['Price']
        .fillna(pd.to_numeric(df['Price']).mean())
        .replace('', str(pd.to_numeric(df['Price']).mean()))
        .astype('float64')
        .round(2)
    )

    df.to_csv('phone_specs_refined.csv', index=False)

    return df


def preprocess_ml(df):
    df_ml = pd.DataFrame()

    df_ml['chipset'] = replace_infreq_values(
        df['Chipset']
        .str.split()
        .str[0]
        .str.lower(),
        threshold=5,
        replacement_value='unpopular chipset')

    df_ml['ram'] = df['RAM']
    df_ml['rear_camera_count'] = df['Rear Camera'].apply(rear_camera_count)
    df_ml['rear_camera_main_mp'] = (
        df['Rear Camera']
        .apply(rear_camera_main_mp)
        .astype('float'))
    df_ml['front_camera_count'] = df['Front Camera'].apply(front_camera_count)
    df_ml['front_camera_main_mp'] = (
        df['Front Camera']
        .apply(front_camera_main_mp)
        .astype('float'))
    df_ml['storage'] = df['Storage']
    df_ml['expansion'] = df['Expansion']    
    df_ml['sim_card'] = df['SIM Card']
    df_ml['cellular'] = df['Cellular'].str.replace('None', '3G HSPA+')
    df_ml['wifi'] = df['Wi-Fi']
    df_ml['nfc'] = df['NFC']
    df_ml['bluetooth'] = df['Bluetooth']
    df_ml['positioning'] = df['Positioning']
    df_ml['usb_otg'] = df['USB OTG']
    df_ml['usb_port'] = df['USB PORT']
    df_ml['sound'] = df['Sound']
    df_ml['fm_radio'] = df['FM Radio']
    df_ml['biometrics'] = df['Biometrics']
    df_ml['sensors'] = df['Sensors']
    df_ml['batt_mah'] = (
        df['Battery']
        .apply(batt_mah)
        .replace(0, 3000))
    df_ml['batt_fast_charging'] = df['Battery'].apply(batt_fast_charging)
    df_ml['material'] = df['Material']
    df_ml['dimensions'] = df['Dimensions']
    df_ml['weight'] = df['Weight']
    df_ml['colors'] = df['Colors']
    df_ml['launch_date'] = df['Launch Date']
    df_ml['price'] = df['Price'].round(0)

    df_ml.to_csv('phone_specs_refined_for_ml_all.csv', index=False)

    filtered_cols = [
        'name',
        'stars_ave',
        'total_votes',
        'likes',
        'screen_diag_in',
        'screen_display',
        'screen_reso_1',
        'screen_reso_2',
        'screen_density',
        'os',
        'chipset',
        'cpu_score',
        'gpu_score',
        'ram',
        'rear_camera_count',
        'rear_camera_main_mp',
        'front_camera_count',
        'front_camera_main_mp',
        'storage',
        'cellular',
        'batt_mah',
        'batt_fast_charging',
        'weight',
        'launch_date',
        'price',
    ]

    df_ml2 = df_ml[filtered_cols]
    df_ml2.to_csv('phone_specs_refined_for_ml.csv', index=False)
    
def main():
    df = pd.read_csv('phone_specs_raw.csv')
    preprocess_ml(preprocess(df))


if __name__ == '__main__':
    main()

from phone_specs_scraper import get_full_specs_for_phone
from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime

datetime.now().strftime('%Y-%m-%d %H:%M:%S')

get_full_specs_for_phone("https://www.pinoytechnoguide.com/smartphones/oppo-a18")

html = 'https://www.pinoytechnoguide.com/smartphones/oppo-a18'
soup = BeautifulSoup(requests.get(html).text, 'lxml')

spec_list = (
    soup
    .find('tbody')
    .find_all('tr')
)

for spec in spec_list:
    # print(spec.text)

    spec_list = list(filter(None, spec))
    print(spec_list)


data = pd.read_csv('phone_specs_raw.csv')
data.sort_values('timestamp').head(5)

date_strs = ["June 30, 2021", "September 2022"]

def unify_date_format(date_str):
    # Try parsing with day included
    try:
        return datetime.strptime(date_str, "%B %d, %Y")
    except ValueError:
        # Parsing failed, try parsing without day, defaulting to the first of the month
        try:
            return datetime.strptime(date_str, "%B %Y").replace(day=1)
        except ValueError as e:
            # Handle unexpected formats or other parsing issues
            print(f"Error parsing date: {e}")
            return None
        
date_strs = ["June 30, 2021", "September 2022"]

unified_dates = [unify_date_format(date_str) for date_str in date_strs]
for original, unified in zip(date_strs, unified_dates):
    print(f"Original: {original}, Unified: {unified.strftime('%B %d, %Y')}")


x = [None, 2, 3]

list(filter(None, x))



from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime
import time

html = 'https://www.pinoytechnoguide.com/smartphones/sony-xperia-xa1'


soup = BeautifulSoup(requests.get(html).text, 'lxml')
spec_dict = {}
spec_list = (
    soup
    .find('tbody')
    .find_all('tr')
)
    
try:
    spec_dict['stars'] = soup.find('span', {'id': 'ave-stars'}).text
    spec_dict['stars_count'] = soup.find('span', {'id': 'count-stars'}).text
    spec_dict['like_share'] = soup.find('div', {'class':'grid3 grid_2'}).a.text
    spec_dict['price'] = (
        soup.find_all('div', {'class':'grid3'})[2]
        .text
        .replace('\t', '')
        .replace('\r', '')
        .replace('\n', '')
        .split(' ')[0]
    )
except:
    spec_dict['stars'] = None
    spec_dict['stars_count'] = None
    spec_dict['like_share'] = None


soup.find('div', {'class':'phone_grid'}).find_all('div', {'class':'grid3'})[2].div

soup.find('div', {'id': 'specsv2-image-container'}).a['href']



############################
## analysis and cleaning
############################

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_csv('phone_specs_raw.csv')

x=df['cpu']

