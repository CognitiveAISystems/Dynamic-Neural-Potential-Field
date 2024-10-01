import requests
from urllib.parse import urlencode
import zipfile


def main():
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://disk.yandex.ru/d/xLeW_jrUpTVnCA' 

    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']
    print('downloading...')
    download_response = requests.get(download_url)
    with open('TransPath_data.zip', 'wb') as f:
        f.write(download_response.content)
    print('extracting...')
    with zipfile.ZipFile('TransPath_data.zip', 'r') as zip_ref:
        zip_ref.extractall('./TransPath_data')
    print('done!')

if __name__ == '__main__':
    main()
