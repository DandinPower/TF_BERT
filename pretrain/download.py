import os
import requests
import zipfile
import hashlib
import tarfile
import sys

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['bert.base'] = (DATA_URL + 'bert.base.torch.zip','225d66f04cae318b841a13d32af3acc165f253ac')
DATA_HUB['bert.small'] = (DATA_URL + 'bert.small.torch.zip','c72329e68a732bef0452e4b96a1c341c8910f81f')
DATA_HUB['reviews'] = ('https://doc-0c-3c-docs.googleusercontent.com/docs/securesc/q1dvmbc1qiv8l7atfei4dn8hmjjurb04/sknvaftvm22ab0pgd7rk498cpa4v3l3l/1646325600000/18016463916621271196/18016463916621271196/1-X-dr2upJRaC-F8KeZ-xWbL5k2x1_gbo?e=download&ax=ACxEAsZzRClgCM0QIqBP7EwY_eGDm7FcHXS1f4es7BVxjw2oywb6ssajtNMs138RxhHN4w6mLkLU00W76bbCDUMZZb2Uth-MzPZa2zOO06yX1ggn-3W7RT70KvlHr5JILWay0B_GMJb8WyNUf3Jz1Ke18OrzxT3BKnt1yqnjgLstKFAAIrcmFzyDrK4JjPCSTdBDj3f7uBu4W-Shb9imzR_X-1AALq2HEXyB3xBr_K9B5RlxFDk4B1lLY2IB_JOLx_wqpC8iWaqAlq1iHOKgZtD_yHqlDJd2XSq-pnVmJZ9ZDlMS-GWcdcyJPDUgD9XgB1YeMop3JLZNkR8Q7dgJEE_jg-UwR3JhZPbcHVQepR0ftK4egPFp1s6Rhb_CKlNflOH6Asc1g5Z9w1VFU3-kZWTWBNk5xf5ihPq25dB3XSkuWpo7tM5MypS2Y58xgcDKDTAqxBqjosatV-45mHkNsZzDxIEe5lMibkjLJG89WTfGtzg3TBtyjan6dpYlL2t9GXR_xHlakjV9XqdJf8EyAOJdXZMgKvNbvveQaFXyPTx9fk8vepZUKVHrWIF4O3QFSZ-yBCSM2ZMaO51grz15z-SLsYiyw3XfQV80qHxuHHtJcJhSv5pZ2bcudCJgSc9uNoY2t4lM7PQgWH1NfhOgx8nposz2AP4scz6B3DrOlw4&authuser=1&nonce=0o3f3407vacce&user=18016463916621271196&hash=sc8ekf5rgvumos3b91sln5p3kdo836j4','fba480ffa8aa7e0febbb511d181409f899b9baa5')

def download(name, cache_dir=os.path.join('data')):
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

download_extract("bert.small")
download_extract("bert.base")
#download_extract("reviews")