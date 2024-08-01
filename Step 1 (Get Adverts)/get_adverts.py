import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
import random
import winsound


pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)


def get_page_source(url):
    """
    This Function use to get the content of the url
    Args:
        url: Website adress

    Returns: HTML content

    """
    r = requests.get(url, headers = headers)
    soup = BeautifulSoup(r.content, 'html.parser')
    return soup


headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/"
                  "537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }


def get_links_list(url):
    """
    This Function provides the links of subpages
    Args:
        url: Website adress

    Returns: List of links

    """
    r = requests.get(url, headers = headers)
    soup = BeautifulSoup(r.content, 'html.parser')
    links = soup.find_all('a', attrs = {'class': 'link-overlay'})
    links_set = set()

    for link_ in links:
        href = link_.get('href')
        links_set.add(href)

    return list(links_set)


# columns = ['marka', 'seri', 'model', 'yil', 'kilometre', 'Fiyat', 'vites_tipi', 'yakit_tipi', 'kasa_tipi', 'renk',
#            'motor_hacmi', 'motor_gucu', 'cekis', 'arac_durumu', 'ort._yakit_tuketimi', 'yakit_deposu', 'boya-degisen',
#            'Detail_CarSegment', 'ilan_tarihi', 'takasa_uygun', 'il', 'kimden', 'ilan_no']
car_df = pd.DataFrame()

counter = 0

# Çoklu sayfalar için:
for page in range(1, 51):
    url = 'https://www.arabam.com/ikinci-el/otomobil/volkswagen-polo?take=50' + '&page={}'.format(page)
    links_list = get_links_list(url)

    for link in links_list:
        full_url = 'https://www.arabam.com' + link
        content = get_page_source(full_url)
        all_scripts = content.find_all('script', attrs = {'type': 'text/javascript'})

        if all_scripts:
            script_info = all_scripts[-1]
            cd_info_dict = dict(re.findall(r"'CD_([^']+)' *: *'([^']+)'", script_info.text))

            # Elde edilen değerleri df'e ekleme:
            car_df = pd.concat([car_df, pd.DataFrame(cd_info_dict, index = [0])], ignore_index = True)
            print('Yeni satır(ilan) df\'e eklendi. Sayaç: {}' .format(counter))
            counter += 1

            # # Rastgele bekleme süresi
            # sleep_time = random.uniform(1, 5)
            # time.sleep(sleep_time)
        else:
            print('Script bulunamadı!!!')
    print("\n\n{}" .format(url))
    print("############# {}.SAYFA TAMAMLANDI #############\n\n" .format(page))

winsound.Beep(1000, 2000)


# Tekli sayfalar için:
url = 'https://www.arabam.com/ikinci-el/otomobil/volkswagen-new-beetle?take=50'
links_list = get_links_list(url)

for link in links_list:
    full_url = 'https://www.arabam.com' + link
    content = get_page_source(full_url)
    all_scripts = content.find_all('script', attrs = {'type': 'text/javascript'})

    if all_scripts:
        script_info = all_scripts[-1]
        cd_info_dict = dict(re.findall(r"'CD_([^']+)' *: *'([^']+)'", script_info.text))

        # Elde edilen değerleri df'e ekleme:
        car_df = pd.concat([car_df, pd.DataFrame(cd_info_dict, index = [0])], ignore_index = True)
        print('Yeni satır(ilan) df\'e eklendi. Sayaç: {}' .format(counter))
        counter += 1

        # # Rastgele bekleme süresi
        # sleep_time = random.uniform(1, 5)
        # time.sleep(sleep_time)
    else:
        print('No script')


# df'yi kaydet:
car_df.to_csv('car_1.csv', index = False, mode = 'a')

