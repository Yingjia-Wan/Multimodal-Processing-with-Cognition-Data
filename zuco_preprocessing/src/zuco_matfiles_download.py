import io
import requests

# 1. Prepare the subject OSF file URL
## SC
subject_list = ["ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZJS", "ZKB", "ZKH", "ZKW", "ZMB", "ZPH"] # SC, task1_NR, 12 subjects
url_list = ['https://osf.io/download/kpn7d/', # ZAB
            'https://osf.io/download/kmxed/',
            'https://osf.io/download/v6np3/',
            'https://osf.io/download/e2ycq/', #ZGW
            'https://osf.io/download/p69s8/',
            'https://osf.io/download/5zxuc/', #ZJN
            'https://osf.io/download/mg6by/', #ZJS
            'https://osf.io/download/nvy84/', #ZKB
            'https://osf.io/download/6v5ad/',
            'https://osf.io/download/pnbxc/', #ZKW
            'https://osf.io/download/2q9an/', #ZMB
            'https://osf.io/download/jkqrm/', #ZPH
            ]

subject_urls_dict = {}
for i, subject in enumerate(subject_list):
  subject_urls_dict[subject] = url_list[i]
print('Subject-file-urls:','\n', subject_urls_dict) # make sure the subject-url is correcly aligned.


# 2. Download the MATLAB file

for SUBJECT_NAME, osf_file_url in subject_urls_dict.items():
  # Define the local file path
  # SC
  local_file_path = f'../../zuco-nlp/sentiment-analysis/Data_to_preprocess/result{SUBJECT_NAME}_SR.mat'
  # RC
  # local_file_path = f'../../zuco-nlp/relation-classification/Data_to_preprocess/result{SUBJECT_NAME}_NR.mat'

  # Download the MATLAB file
  r = requests.get(osf_file_url)
  if r.status_code != 200:
    print(f'Failed to download the {SUBJECT_NAME} MATLAB file')
  else:
    with open(local_file_path, 'wb') as f:
      f.write(r.content)
    print(f'{SUBJECT_NAME} MATLAB file downloaded and saved to:', local_file_path)

# Note: It's easier to download the individual files because local_file_path requires to be a file path, not a directory path.:
# ZAB file: https://osf.io/download/kpn7d/
# ZDM file: https://osf.io/download/kmxed/
# ZDN file: https://osf.io/download/v6np3/
# ....