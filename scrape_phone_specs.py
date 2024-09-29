from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime
import time

def get_full_specs_for_phone(html):
    '''
    Scrapes for the needed specs for each article of the Phone and stores it
    inside a dictionary.
    '''
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
        spec_dict['pic_link'] = soup.find('div', {'id': 'specsv2-image-container'}).a['href']
    except:
        spec_dict['stars'] = None
        spec_dict['stars_count'] = None
        spec_dict['like_share'] = None
        spec_dict['price'] = None
        spec_dict['pic_link'] = None

    for spec in spec_list:
        try:
            raw_specs = (
                spec
                .text
                .replace('\t', '')
                .replace('\r', '')
                .split('\n')
            )
            spec_list = list(filter(None, raw_specs))
            
            # Special considerations.
            # Some data has Camera instead of Rear Camera or Networks instead 
            # Cellular. Also skip TV, GPS, and Buy Online as only a few rows
            # have data on these.
            if spec_list[0].lower().strip() == 'camera':
                spec_list[0] = 'Rear Camera'
            if spec_list[0].lower().strip() == 'networks':
                spec_list[0] = 'Cellular'
            if spec_list[0].lower().strip() == 'screen':
                spec_list[0] = 'Display'
            if spec_list[0].lower().strip() == 'launch date':
                spec_list[0] = 'Release Date'
            if spec_list[0].lower().strip() == 'price':
                spec_list[0] = 'price'                
            if spec_list[0].lower() in ['tv', 'gps', 'buy online']:
                continue
            spec_dict[spec_list[0]] = spec_list[1]
        except:    
            pass
        
    # Add time stamp
    spec_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return spec_dict


def specs_list(base_html='https://www.pinoytechnoguide.com/smartphones/page/'):
    try:
        page_num = 1
        is_last_page = False
        dne_text = 'does not exist'
        phone_spec_list = []
        
        while not is_last_page:
            # html should be https://www.pinoytechnoguide.com/smartphones/page/
            html = f'{base_html}{str(page_num)}'
            html_text = requests.get(html).text
            soup = BeautifulSoup(html_text, 'lxml')
            page_main_text = soup.find('div', {'id': 'main'}).text
            
            # stops while loop when it reaches last page.
            if dne_text in page_main_text:
                is_last_page = True
                break
                
            print(f'Current page: {page_num}')
            total_time = 0
            phones = soup.find_all('tr', {'class':'phone_block'})
            
            for phone in phones:
                t1 = time.time()

                phone_spec = {}
                name = phone.a.text.strip()
                link = phone.a['href']
                
                phone_spec['name'] = name
                phone_spec['link'] = link
                
                # Fills up dictionary with all data coming from phone's article
                phone_spec.update(get_full_specs_for_phone(link)) 
                phone_spec_list.append(phone_spec)
                
                t2 = time.time()
                time_per_phone = t2 - t1
                print(f'{name:<60}: {time_per_phone:.3f}')
                total_time += time_per_phone
                
            print(f'Page {page_num} time: {total_time:.2f} sec')
            print('')
            page_num += 1
             
        print("---------DONE---------")
        
        return phone_spec_list
                  
    except Exception as e:
        print(e)
        return
    
def main(): 
    t1 = time.time()
    df = pd.DataFrame(specs_list())
    t2 = time.time()
    total_time = t2 - t1
    print(f'Over All time: {total_time:.2f} sec --> {total_time/60:.1f} mins')
    df.to_csv('phone_specs_raw.csv', index=False)

if __name__ == '__main__':
    main()
