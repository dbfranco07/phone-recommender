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
    'June 20, 2024 (announced on June 12)': 'June 20, 2024',
    'October 17 2018': 'October 17, 2018' 
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
    'USD': 58.8,
    '￥': 0.38,
    'INR': 0.68,
    'EUR': 60.64,
    'NT$': 1.80,
    'CNY': 7.79,
    'RUB': 0.59,
    'IDR': 0.0037,
    'GBP': 69.56,
    'BRL': 11.51,
    'KSH': 0.3914,
}

missing_price_dict = {
    'vivo Y27 5G': '4790',
    'OPPO Reno8 Pro 5G': '2370',
    'vivo Y72 5G': '27400',
    'realme X50 Pro 5G': '22700',
    'Vivo V19': '17611',
    'Nokia C2': '23520',
    'Motorola Razr': '$450',
    'ASUS Zenfone Live L2': '$600',
    'LG V50 ThinQ 5G': 'EUR550',
    'Samsung Galaxy A8s': 'EUR400',
    'Vivo Y93': 'EUR200',
    'Samsung Galaxy J4 Core': 'EUR150',
    'Sony Xperia XZ3': 'EUR510',
    'LG G7 One': 'EUR350',
    'LG G7 Fit': 'EUR200',
    'LG Q Stylus': 'EUR300',
    'LG Q7': 'EUR150',
    'Huawei Y3 2018': 'EUR90',
    'Sony Xperia XZ2 Premium': 'EUR460',
    'OPPO A1': 'EUR270',
    'Sony Xperia XZ2 Compact': 'EUR430',
    'Sony Xperia XZ2': 'EUR360',
    'Nokia 8 Sirocco': 'EUR1010',
    'Alcatel 3C': 'EUR130',
    'HTC U11 EYEs': 'EUR300',
    'Sony Xperia XA2': 'EUR200',
    'Cherry Mobile Flare S6 Premium': '4000',
    'Xiaomi Redmi Note 5A Prime': 'EUR140',
    'Nokia 7': 'EUR220',
    'Apple iPhone 8 Plus': 'EUR340',
    'Apple iPhone 8': 'EUR210',
    'Samsung Galaxy J5 (2017)': 'EUR130',
    'Essential Phone': 'EUR330',
    'Samsung Galaxy J3 2017': 'EUR110',
    'BlackBerry Aurora': 'EUR200',
    'Samsung Galaxy Xcover 4': 'EUR240',
    'BlackBerry KEYone': 'EUR290',
    'LG X Power2': 'EUR180',
    'Blackview BV7000': '$144',
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

gpu_score_df= pd.read_csv(
    'gpu_score.csv', 
    header=0, 
)
gpu_manual_score_df = pd.DataFrame(
    list(gpu_score_update.items()),
    columns=['GPU', 'gpu_score']
)
gpu_score_df2 = pd.concat(
    [gpu_score_df, gpu_manual_score_df]
)

#######################################################################
# Utility Functions
#######################################################################
def convert_price(value):
    '''Converts price to Philippine Peso for easier comparison.'''
    if value == None or len(value)==0:
        return 0
    elif value[:3] in conversion_factor:
        return int(float(value[3:]) * conversion_factor[value[:3]])
    elif value[0] == '$':
        return int(float(value[1:]) * conversion_factor['USD'])
    else:
        return int(value)
    
# preprocessing functions
def screen_diag_in(screen_data):
    '''Gets the diagonal length of a scrin in inches.'''
    pattern = r'(\d+(?:\.\d+)?)-inch'
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
    
def screen_display(screen_data):
    '''Gets the screen display (e.g. AMOLED)'''
    pattern = r'inch\s(.*?)\sDisplay'
    match = re.search(pattern, screen_data)
    if match:
        if 'lcd' in match.group(1).lower():
            return 'lcd'
        if 'oled' in match.group(1).lower():
            return 'oled'
        if 'ltps' in match.group(1).lower():
            return 'ltps'        
        else:
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

@np.vectorize
def screen_density(screen_data, diag, reso_width, reso_length):
    '''Gets the density in pixel per inch.'''
    pattern = r'(\d+)\s*ppi'
    match = re.search(pattern, screen_data)
    if match:
        return int(match.group(1))
    else:
        return int(np.floor(np.sqrt(reso_width**2 + reso_length**2) / diag))
    
def screen_refreshrate(screen_data):
    '''Gets the refresh rate of the screen.'''
    pattern = r'(\d+)\s*(Hz)'
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
    
def os_category(os):
    if 'ios' in os.lower():
        return 'ios'
    elif 'android' in os.lower():
        return 'android'
    else:
        return os.split(' ')[0].lower()
    
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
    
def batt_mah(batt_data):
    '''Gets the mAh of battery of '''
    pattern = r'(\d+(\.\d+)?)\s*(mah)'
    match = re.search(pattern, batt_data, re.IGNORECASE)
    if match:
        return int(match.group(1))
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
    
def unify_date(date):
    try:
        u_date = pd.to_datetime(date, format='%B %d, %Y', errors='coerce')
        if pd.isnull(u_date):
            u_date = pd.to_datetime(date, format='%b-%y', errors='coerce')
            if pd.isnull(u_date):
                return None
        return u_date
    except:
        return None
#######################################################################
# Cleaning
#######################################################################
def preprocess(df):
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

    df['screen_diag_in'] = df['Display'].apply(screen_diag_in)

    df['screen_display_tech'] = df['Display'].apply(screen_display)

    df['screen_reso_width'] = df['Display'].apply(screen_reso, group_num=1)

    df['screen_reso_length'] = df['Display'].apply(screen_reso, group_num=2)

    df['screen_ppi'] = screen_density(
        df['Display'], 
        df['screen_diag_in'], 
        df['screen_reso_width'], 
        df['screen_reso_length']
    )

    df['refreshrate'] = df['Display'].apply(screen_refreshrate)

    df['rear_camera_count'] = df['Rear Camera'].apply(rear_camera_count)

    df['rear_camera_main_mp'] = (
        df['Rear Camera']
        .apply(rear_camera_main_mp)
        .astype('float')
    )
    
    df['front_camera_count'] = df['Front Camera'].apply(front_camera_count)

    df['front_camera_main_mp'] = (
        df['Front Camera']
        .apply(front_camera_main_mp)
        .astype('float')
    )
    
    df['os'] = (
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
        .str.replace('14.1', '14')
        .str.replace('14.2', '14')
        .str.split('with')
        .str[0]
        .str.strip()
        .apply(os_category)
    )

    df['cpu_score'] = (
        df['CPU']
        .fillna('Hexa-core (2x high power Lightning cores at 2.66 GHz + 4x low power Thunder cores at 1.82 GHz)')
        .str.replace('GHz + 4', 'GHz & 4')
        .str.replace('Performance Cores', '2GHz')
        .str.replace('Efficiency Cores', '1.5GHz')
        .apply(calculate_performance_score, minimum=4)
    )

    df['gpu'] = (
        df['GPU']
        .fillna('Unknown')
        .str.strip()
        .str.replace('_', 'Unknown')
    )


    df[['gpu', 'gpu_score']] = (
        df[['GPU']]
        .set_index('GPU')
        .merge(gpu_score_df2, how='left', on='GPU')
        .fillna(50.0)
    )

    df['ram'] = (
        df['RAM']
        .str.replace('G', ' ')
        .str.replace(',', '')
        .str.replace('512', '0.512')
        .str.split()
        .str[0]
        .astype('float')
    )

    df['storage'] = (
        df['Storage']
        .str.replace('G', ' ')
        .str.split()
        .str[0]
        .astype('int32')
    )

    df['cellular'] = df['Cellular'].str[:2].str.replace('G', 'g')

    df['batt_mah'] = (
        df['Battery']
        .apply(batt_mah)
        .replace(0, 3000)
    )

    df['batt_fast_charging'] = df['Battery'].apply(batt_fast_charging)

    df['release_date'] = (
        df['Release Date']
        .ffill()
        .replace(replace_launchdate)
        .apply(unify_date)
    )

    df['release_date_month'] = df['release_date'].dt.month.astype(int)

    df['release_date_year'] = df['release_date'].dt.year.astype(int)

    features = [
        'name',
        'stars',
        'stars_count',
        'like_share',
        'brand',
        'screen_diag_in',
        'screen_display_tech',
        'screen_reso_width',
        'screen_reso_length',
        'screen_ppi',
        'refreshrate',
        'rear_camera_count',
        'rear_camera_main_mp',
        'front_camera_count',
        'front_camera_main_mp',
        'os',
        'cpu_score',
        'gpu_score',
        'ram',
        'storage',
        'cellular',
        'batt_mah',
        'batt_fast_charging',
        'release_date_month',
        'release_date_year',
        'price',
    ]

    df_new = df[features]

    return df_new

if __name__ == '__main__':
    df = pd.read_csv('phone_specs_raw.csv')
    df_new = preprocess(df)
    df_new.to_csv('phone_specs_processed.csv', index=False)