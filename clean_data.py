import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.preprocessing import OneHotEncoder

#######################################################################
# Replacement Dictionaries
#######################################################################
replace_biometrics = {
    'Face Recognition & Fingerprint Sensor (side mounted)': 'Face Recognition & Fingerprint Sensor (side-mounted)',
    'Face Recognition & Fingerprint Sensor (Side Mounted)': 'Face Recognition & Fingerprint Sensor (side-mounted)',
    'Face Recognition & Fingerprint Scanner (side-mounted)':'Face Recognition & Fingerprint Sensor (side-mounted)', 
    'Face Recognition & Fingerprint Sensor (side mounted': 'Face Recognition & Fingerprint Sensor (side-mounted)',    
    'Face Recognition & Fingerprint Sensor (under display)': 'Face Recognition & Fingerprint Sensor (under-display)',
    'Face Recognition & Fingerprint Sensor (Under Display)': 'Face Recognition & Fingerprint Sensor (under-display)',
    'Face Recognition & Fingerprint Scanner (under display)': 'Face Recognition & Fingerprint Sensor (under-display)',
    'Face Recognition & Fingerprint Sensor (under-dispaly)': 'Face Recognition & Fingerprint Sensor (under-display)',
    'Face Recognition & Fingeprint Sensor (Under Display)': 'Face Recognition & Fingerprint Sensor (under-display)',
    'Face Recognition & Fingerprint Scanner (in-display)': 'Face Recognition & Fingerprint Sensor (in-display)',    
    ', Heart Rate Sensor, Iris Scanner & Fingerprint Sensor': 'Heart Rate Sensor, Iris Scanner & Fingerprint Sensor',
    ', Heart Rate Sensor & Fingerprint Sensor': 'Heart Rate Sensor & Fingerprint Sensor',
    'Fingerprint Sensor (side mounted)': 'Fingerprint Sensor (side-mounted)',
    'Fingerprint Sensor (under display)': 'Fingerprint Sensor (under-display)',
    'Fingerprint Sensor (Under display)': 'Fingerprint Sensor (under-display)',
    'Fingeprint Sensor (Under Display)': 'Fingerprint Sensor (under-display)',
    'Fingerprint Sensor (Under Display)': 'Fingerprint Sensor (under-display)',
    'Fingeprint Sensor (under-display)': 'Fingerprint Sensor (under-display)',
    'Face Recognition & Fingerprint Sensor (Side)': 'Face Recognition & Fingerprint Sensor (side-mounted)',
    'Face Recognition & Fingerprint Sensor (Rear)': 'Face Recognition & Fingerprint Sensor (rear)',
    'Face Recognition, Heart Rate Sensor & Fingerprint Sensor (side mounted)': 'Face Recognition, Heart Rate Sensor & Fingerprint Sensor (side-mounted)',
    'Fingerprint Sensor (Side Mounted)': 'Fingerprint Sensor (side-mounted)',    
    'FaceID Face Recognition': 'Face Recognition',
    'Face Recognition, Heart Rate Sensor, Iris Scanner & Fingerprint Sensor (Under Display, Ultrasonic)': 'Face Recognition, Heart Rate Sensor, Iris Scanner & Fingerprint Sensor (under-display, ultrasonic)',
}

replace_launchdate = {
    'Ootober 23, 2021': 'October 23, 2021',
    'September 15, 2021 (Philippines)': 'September 15, 2021',
    'May 29, 2020 - Release Date in the Philippines': 'May 29, 2020',
    'Not yet official.': 'December 31, 2017',
}

replacement_expansion = {
    'Expandable up to 256GBGB via NanoMemory Card': 'Expandable up to 256GB via NanoMemory Card',
    'Expandable up to 512GBGB via NanoMemory Card': 'Expandable up to 512GB via NanoMemory Card',
}

replacement_sim_card = {
    'Dual SIM': 'Dual SIM (Regular)',
    'Single SIM': 'Single SIM (Regular)',
    'Dual SIM (Micro-SIM)': 'Dual SIM (Micro)',
    'Dual SIM (Nano-SIM)': 'Dual SIM (Nano)',
    'Single SIM (Micro-SIM)': 'Single SIM (Micro)',
    'Single SIM (Nano-SIM)': 'Single SIM (Nano)',
    'Triple SIM (Nano-SIM)': 'Triple SIM (Nano)',
}

replacement_cellular = {
    ' ': 'None'
}

replacement_wifi = {
    '6E': '6e',
    'Dual': 'dual',
    'Band': 'band',
}

conversion_factor = {
    'AED': 15.4,
    'USD': 50,
    '￥': 0.38,
    'INR': 0.68,
    'EUR': 60.64,
    'NT$': 1.76,
    'CNY': 7.79,
    'RUB': 0.59,
    'IDR': 0.0037,
    'GBP': 69.56,
    'BRL': 11.51,
    'KSH': 0.3914
}

gpu_score_update = {
    'Mali-400': 4.05,
    'Mali-G57 MC2': 107.5,
    'Mali-G52 MC2': 87.0,
    'PowerVR GE8322': 19.0,
    'Mali-G68 MC4': 119.0,
    'Mali-G76 MC4': 151.5,
    'Mali-G57': 107.5,
    'Mali-G71': 105.0,
    'Mali-G76': 96.5,
    'Mali-G52': 87.0,
    'Mali-T820': 60.0,
    'Mali-G72': 38.0,
    'Mali-G77 MC9': 184.5,
    'Mali-G710 MC10': 294.0,
    '5-core Apple GPU': 430,
    'Mali-T860': 17.0,
    'Mali-T880': 82.0,
    'Mali-T830': 34.0, 
    'Mali-G51': 39.0,
    'Mali-G610 MC6': 272.0,
    'IMG PowerVR GE8320': 20.0,
    'Mali-G77': 199.0,
    'Apple GPU (4 cores)': 395.75,
    '4-core Apple GPU': 395.75,
    'Apple GPU': 200.0,
    '3-Core GPU': 200.0,
    'Apple 6-core GPU': 500.0,
    'Mali-G57 MC3': 107.5,
    'Mali-T720 MP3': 12.0,
}

#######################################################################
# Utiliti Functions
#######################################################################
def convert_price(value):
    '''Converts price to Philippine Peso for easier comparison.'''
    if len(value) == 1:
        return value[0]
    else:
        return conversion_factor[value[0]] * float(value[1])
    
# preprocessing functions
def screen_diag_in(screen_data):
    '''Gets the diagonal length of a scrin in inches.'''
    pattern = r'(\d+\.\d+)-inch'
    match = re.search(pattern, screen_data)
    if match:
        return float(match.group(1))
    else:
        return 0
    
def screen_display(screen_data):
    '''Gets the screen display (e.g. AMOLED)'''
    pattern = r'inch\s(.*?)\sDisplay'
    match = re.search(pattern, screen_data)
    if match:
        return match.group(1)
    else:
        return 'None'

def screen_reso(screen_data, group_num):
    '''Gets the resolution pixels.'''
    pattern = r'\((\d+)\s*x\s*(\d+)\s*Pixels'
    match = re.search(pattern, screen_data, re.IGNORECASE)
    if match:
        return int(match.group(group_num))
    else:
        return 0
    
def screen_density(screen_data):
    '''Gets the density in pixel per inch.'''
    pattern = r'(\d+)\s*ppi'
    match = re.search(pattern, screen_data)
    if match:
        return int(match.group(1))
    else:
        return 0
    
def screen_refreshrate(screen_data):
    '''Gets the refresh rate of the screen.'''
    pattern = r'(\d+)\s*(Hz Refresh Rate)'
    match = re.search(pattern, screen_data, re.IGNORECASE)
    if match:
        return int(match.group(1))
    else:
        return 0

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
    df['like_share'] = (
        df['like_share']
        .str.split()
        .str[0]
        .str.replace(',', '')
        .fillna(0)
        .astype('int')
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

    # Manual cleaning for RAM
    df['RAM'] = (
        df['RAM']
        .str.replace('G', ' ')
        .str.replace(',', '')
        .str.replace('512', '0.512')
        .str.split()
        .str[0]
        .astype('float')
    )

    # Manual cleaning for Storage
    df['Storage'] = (
        df['Storage']
        .str.replace('G', ' ')
        .str.split()
        .str[0]
        .astype('int32')
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
    df_ml['name'] = df['Name']
    df_ml['stars_ave'] = df['Stars']
    df_ml['total_votes'] = df['Stars Count']
    df_ml['likes'] = df['LikeShare']
    df_ml['screen_diag_in'] = df['Screen'].apply(screen_diag_in)
    df_ml['screen_display'] = replace_infreq_values(
        df['Screen'].apply(screen_display),
        threshold=10,
        replacement_value='Onpopular Screen Display')
    df_ml['screen_reso_1'] = df['Screen'].apply(screen_reso, group_num=1)
    df_ml['screen_reso_2'] = df['Screen'].apply(screen_reso, group_num=2))   
    df_ml['screen_density'] = df['Screen'].apply(screen_density)
    df_ml['os'] = (
        df['OS']
        .str.split('(')
        .str[0]
        .str.strip()
        .str.replace('Android 5.1 Lollipop', 'Android 5')
        .str.replace('6.0.1', '6')         
        .str.replace('Android 6 Marshmallow', 'Android 6')        
        .str.replace('Android 6.0 Marshmallow', 'Android 6')
        .str.replace('7.1.1', '7')
        .str.replace('7.1.2', '7')
        .str.replace('7.1', '7')   
        .str.replace('Android 7 Nougat', 'Android 7')
        .str.replace('Android 7.0 Nougat', 'Android 7')                 
        .str.replace('8.1 Oreo', '8')
        .str.replace('Android 8.0 Oreo', 'Android 8')
        .str.replace('Android Oreo', 'Android 8')        
        .str.replace('Android 9.0 Pie', 'Android 9')
        .str.replace('Android 10.0 Pie', 'Android 10')
        .str.replace('Android 11', 'Android 11') 
        .str.replace('Android 12', 'Android 12')               
        .str.replace('EMUI 13.1', 'EMUI 13')
        .str.replace('14.1', '14'))
    df_ml['chipset'] = replace_infreq_values(
        df['Chipset']
        .str.split()
        .str[0]
        .str.lower(),
        threshold=5,
        replacement_value='unpopular chipset')
    df_ml['cpu'] = df['CPU']
    df_ml['cpu_score'] = (
        df['CPU']
        .str.replace('GHz + 4', 'GHz & 4')
        .str.replace('Performance Cores', '2GHz')
        .str.replace('Efficiency Cores', '1.5GHz')
        .apply(calculate_performance_score, minimum=4))
    df_ml['gpu'] = df['GPU']
    df_ml[['gpu', 'gpu_score']] = (
        df_ml[['gpu']]
        .set_index('gpu')
        .merge(gpu_score_df2, how='left', on='gpu')
        .fillna(50.0))
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