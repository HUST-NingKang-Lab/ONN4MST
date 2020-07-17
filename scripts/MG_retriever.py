import urllib.request, json
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Download all otu tsv file involved in specific study')
parser.add_argument('-s', '--studyid', type=str, default='MGYS00001601', help='MGYS...')
args = parser.parse_args()

study_id = args.studyid
proxy = urllib.request.ProxyHandler({'http': '127.0.0.1:4780', 'https': '127.0.0.1:4780'})
opener = urllib.request.build_opener(proxy)
urllib.request.install_opener(opener)
# ---------------------------
def getjson_asdict(url):
    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode())
    return data


def retrv_ifotutsv(download_obj):
    tsv_format = {'compression': False, 'name': 'TSV', 'extension': 'tsv'}
    if download_obj['attributes']['file-format'] == tsv_format:
        tsv_link = download_obj['links']['self']
        try:
            urllib.request.urlretrieve(tsv_link, filename=download_obj['id'])
        except urllib.error.ContentTooShortError:
            pass

study_url = 'https://www.ebi.ac.uk/metagenomics/api/v1/studies/' + study_id
study_obj = getjson_asdict(study_url)

analyses_url = study_obj['data']['relationships']['analyses']['links']['related']
analyses_obj = getjson_asdict(analyses_url)

for analysis in analyses_obj['data']:
    downloads_url = analysis['relationships']['downloads']['links']['related']
    downloads_obj = getjson_asdict(downloads_url)
    for file in tqdm(downloads_obj['data']):
        retrv_ifotutsv(file)

