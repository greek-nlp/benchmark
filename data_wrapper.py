import os
import json
import shutil
import hashlib
import tarfile
import zipfile
# !pip install zenodo-get
import zenodo_get 
import numpy as np
import pandas as pd
import seaborn as sns
from functools import reduce
import matplotlib.pyplot as plt
# !pip install datasets
from datasets import load_dataset
import xml.etree.ElementTree as ET
# !pip install wget
import wget
# !pip install zenodo-get
import zenodo_get
import subprocess


def wget_download(resource_id, url):
  os.makedirs(str(resource_id), exist_ok=True)
  # Use wget to download the file (as in >> !wget -P {resource_id} {url})
  wget.download(url=url, out=resource_id)


def zenodo_download(resource_id, zenodo_url):
  os.makedirs(str(resource_id), exist_ok=True)
  # as in >> !zenodo_get {zenodo_url}
  zenodo_get.zenodo_get(zenodo_url, output=resource_id)


def huggingface_download(resource_id, to_folder, dataset_name, splits, subsets=[None]):
  """
  Download the data from HuggingFace
  """
  # Create the directory if it does not exist
  os.makedirs(str(resource_id), exist_ok=True)
  df_dict = {}
  for subset in subsets:
    # Load the dataset
    dataset = load_dataset(dataset_name, subset)

    for split in splits:
      # Convert the dataset to a Pandas DataFrame
      df_hg = pd.DataFrame(dataset[split])
      if resource_id == '250': #The Papaloukas dataset
        df_hg = df_hg.rename(columns={'label': subset})

      if len(subsets) > 1:
        df_dict[f"{split}_{subset}"] = df_hg
      else:
        df_dict[f"{split}"] = df_hg

      # Save the DataFrame to a CSV file
      if subset is not None:
        df_hg.to_csv(f'{to_folder}/{resource_id}_{subset}_{split}.csv', index=False)
      else:
        df_hg.to_csv(f'{to_folder}/{resource_id}_{split}.csv', index=False)

  return df_dict

def run_git_command(command, cwd=None):
    result = subprocess.run(command, shell=True, cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip()

def git_sparse_checkout_download(resource_id, repo_url, down_folder, branch, root_dir):
  """
  Download folder containing the data from github repository
  """
  # move to root dirrectory
  os.chdir(root_dir)

  # Install Git (if not already installed) and configure sparse checkout
  # !sudo apt-get install git -y
  run_git_command(f'git init repo_{resource_id}')
  os.chdir(f'repo_{resource_id}')
  run_git_command(f'git remote add -f origin {repo_url}')
  run_git_command(f'git config core.sparseCheckout true')

  # Define the folder to download
  with open('.git/info/sparse-checkout', 'w') as f:
      f.write(down_folder + '\n')

  # Pull the specific folder from the repository
  run_git_command(f'git pull origin {branch}')

  # Verify if the folder has been downloaded
  if os.path.exists(down_folder):
      print(f"Successfully downloaded {down_folder}")
  else:
      print(f"Failed to download {down_folder}. Please check the folder path and branch name.")

  # Move back to the root directory
  os.chdir(root_dir)



class BarzokasDt:

    def __init__(self, datasets, id_=56):
      self.paper_id = id_ # the ID in the shared resource
      self.datasets = datasets
      self.repo_url = self.datasets[self.datasets.paper_id==self.paper_id].URL.iloc[0]
      self.down_folder = 'data/corpora'  # Data folder path within the git repository
      self.branch = "master"
      self.name = "barzokas"
      self.splits = {'train'}
      self.train = self.download()


    def _create_df(self, dataset_folder):
        df_tsv = pd.read_csv(os.path.join(dataset_folder, "metadata.tsv"), sep='\t')
        print(f"tsv records: {df_tsv.shape}")

        data = []
        txt_root_dir = os.path.join(dataset_folder, "text")
        for txt_folder in os.listdir(txt_root_dir):
          for txt_file in os.listdir(os.path.join(txt_root_dir, txt_folder)):
              if txt_file.endswith('.txt'):
                  # Extract the ID from each TXT filename
                  txt_id = os.path.splitext(txt_file)[0]

                  # Read the content of each TXT file
                  with open(os.path.join(txt_root_dir, txt_folder, txt_file), 'r', encoding='utf-8') as f:
                      txt_content = f.read()

                  # Merge the ID, title, and text content into a new DataFrame
                  row = df_tsv[df_tsv['id'] == txt_id]
                  if not row.empty:
                      row_data = row.iloc[0].to_dict()
                      row_data['text'] = txt_content
                      row_data['status'] = txt_folder
                      data.append(row_data)

        # Create the final DataFrame
        df_final = pd.DataFrame(data)
        return df_final

    def download(self):
      git_sparse_checkout_download(self.paper_id, self.repo_url, self.down_folder, self.branch)
      barzokas_df_list = []
      for data_fname in os.listdir(f"{self.paper_id}"):
        dataset_folder = f"{self.paper_id}/{data_fname}"
        df_56 = self._create_df(dataset_folder)
        df_56["publisher"] = data_fname
        barzokas_df_list.append(df_56)

      df = pd.concat(barzokas_df_list)
      return df

    def get(self, split='train'):
      assert split in {'train'}
      return self.train

    def save_to_csv(self, path = './'):
      self.train.to_csv(os.path.join(path, f'{self.name}.csv'), index=False)


class KorreDt:
  def __init__(self, datasets, root_dir, id_=244):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.id==self.resource_id]
      self.name = 'korre'
      # Download data
      self.root_dir = root_dir
      self.repo_url = self.resource.iloc[0].url
      self.down_folder = 'GNC'  # Data folder path within the git repository
      self.branch = "main"
      self.splits = {'train'}
      self.dataset = None
      self.train = self.download()

  def download(self):
      git_sparse_checkout_download(self.resource_id, self.repo_url, self.down_folder, self.branch, self.root_dir)
      path = os.path.join(self.root_dir, f'repo_{self.resource_id}', self.down_folder)      
      # Merge the two annotators dataframes
      df_annA = pd.read_excel(f'{path}/GNC_annotator_A.xlsx')
      df_annA.columns = ["label_annA", "original_text_annA", "corrected_text_annA", "error_description_annA", "error_type_annA", "fluency_annA"]
      df_annB = pd.read_excel(f'{path}/GNC_annotator_B.xlsx')
      df_annB.columns = ["label_annB", "original_text_annB", "corrected_text_annB", "error_description_annB", "error_type_annB", "fluency_annB"]
      df_ann = pd.merge(df_annA, df_annB, left_index=True, right_index=True, how='inner')

      # Original text
      with open(f"{path}/orig.txt", 'r', encoding='utf-8') as file:
          lines = file.readlines()
      df_orig = pd.DataFrame(lines, columns=['original_text'])
      df_orig['original_text'] = df_orig['original_text'].str.strip()
      df_orig.replace('', np.nan, inplace=True)

      # Corrected text
      with open(f"{path}/corr.txt", 'r', encoding='utf-8') as file:
          lines = file.readlines()
      df_corr = pd.DataFrame(lines, columns=['corrected_text'])
      df_corr['corrected_text'] = df_corr['corrected_text'].str.strip()
      df_corr.replace('', np.nan, inplace=True)
      # merge txts
      df_txt = pd.merge(df_orig, df_corr, left_index=True, right_index=True, how='inner')

      # merge the annotations and the txt
      df_gnc = pd.merge(df_txt, df_ann, left_index=True, right_index=True, how='inner')
      df_gnc.drop(columns=['original_text_annA', 'original_text_annB', 'corrected_text_annA', 'corrected_text_annB'], inplace=True)

      # Drop rows where either 'corrected_text' or 'original_text' is NaN
      df_gnc.dropna(subset=['corrected_text', 'original_text'], how='any', inplace=True)
      # keep only the incorrect sentences
      df_gnc = df_gnc.loc[df_gnc['corrected_text'] != df_gnc['original_text']]
      # Convert all columns of type 'object' to 'string'
      df_gnc = df_gnc.astype({col: 'string' for col in df_gnc.select_dtypes(include='object').columns})
      # remove git repository
      shutil.rmtree(os.path.join(self.root_dir, f'repo_{self.resource_id}'))

      return df_gnc


  def get(self, split='train'):
    assert split in {'train'}
    return self.train

  def save_to_csv(self):
    self.train.to_csv(os.path.join(self.root_dir, f'{self.name}.csv'), index=False)



class ZampieriDt:
    def __init__(self, datasets, id_=341):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.paper_id==self.resource_id]
      self.name = 'zampieri'
      # Download data
      self.repo_url = self.resource.iloc[0].URL
      self.splits = ["train", "test"]
      self.dataset = self.download()
      self.train = self.dataset['train']
      self.test = self.dataset['test']

    def download(self, split='train', csv_datasets_folder='./'):
      dataset_name = 'strombergnlp/offenseval_2020'
      subsets = ["gr"]
      df_dict = huggingface_download(self.resource_id, csv_datasets_folder, dataset_name, self.splits, subsets=subsets)
      return df_dict

    def get(self, split='train'):
      assert split in {'train', 'test'}
      return self.dataset[split]

    def save_to_csv(self, path = './'):
      self.train.to_csv(os.path.join(path, f'{self.name}.csv'), index=False)


class ProkopidisDt:
    def __init__(self, datasets, id_=486):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.paper_id==self.resource_id]
      self.repo_url = self.resource.iloc[0].URL
      self.name = 'prokopidis'
      self.splits = {'train'}
      self.train = self.download()
      self.train['text'] = self.train.Greek

    def _generate_checksum(self, text):
      return hashlib.sha256(text.encode()).hexdigest()

    def get(self, split='train'):
      assert split in {'train'}
      return self.train

    def download(self):
      langs_dict = {
          "eng": "English",
          "epo": "Esperanto",
          "fas": "Farsi",
          "fil": "Filipino",
          "fra": "French",
          "heb": "Hebrew",
          "hin": "Hindi",
          "hun": "Hungarian",
          "ind": "Indonesian",
          "ita": "Italian",
          "jpn": "Japanese",
          "khm": "Khmer",
          "kor": "Korean",
          "mkd": "Macedonian",
          "mlg": "Malagasy",
          "mya": "Burmese",
          "nld": "Dutch",
          "ori": "Odia",
          "pol": "Polish",
          "por": "Portuguese",
          "rum": "Romanian",
          "rus": "Russian",
          "spa": "Spanish",
          "sqi": "Albanian",
          "srp": "Serbian",
          "swa": "Swahili",
          "swe": "Swedish",
          "tur": "Turkish",
          "urd": "Urdu",
          "zhs": "Chinese-simplified",
          "zht": "Chinese-traditional"
      }

      for other_lang in langs_dict:
        data_url = f"{self.repo_url}archives/ell-{other_lang}.zip"
        wget_download(self.resource_id, data_url)
        # Unzip
        with zipfile.ZipFile(f"{self.resource_id}/ell-{other_lang}.zip", 'r') as zip_ref:
          zip_ref.extractall(f"{self.resource_id}/ell-{other_lang}")

      pgv_df_list = []

      namespace = {'xml': 'http://www.w3.org/XML/1998/namespace'}

      # Iterate through TMX files in the directory
      for other_lang, other_lang_name in langs_dict.items():
        print(other_lang)
        file_path = f"{self.resource_id}/ell-{other_lang}/pgv/ell-{other_lang}.tmx"
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Initialize lists to store data
        score = []
        source_lang = []
        target_lang = []

        # Iterate through tu elements
        for tu in root.findall('.//tu'):
            score.append(tu.find('.//prop[@type="score"]').text)

            source = tu.find('.//tuv[@xml:lang="ell"]/seg', namespaces=namespace).text
            target = tu.find(f'.//tuv[@xml:lang="{other_lang}"]/seg', namespaces=namespace).text
            source_lang.append(source)
            target_lang.append(target)

        # Create DataFrame
        df_pair = pd.DataFrame({'Greek': source_lang, other_lang_name: target_lang, f"Greek_{other_lang_name}_score": score})
        df_pair['Checksum'] = df_pair[f'Greek'].apply(self._generate_checksum)
        df_pair.drop_duplicates(subset='Checksum', inplace=True)
        pgv_df_list.append(df_pair)

      # Initialize merged DataFrame with the first DataFrame
      df_pgv = pgv_df_list[0]

      # Merge all DataFrames in the list
      for df_pgv_pair in pgv_df_list[1:]:
          df_pgv = pd.merge(df_pgv, df_pgv_pair, on=['Checksum', 'Greek'], how='outer')

      df_pgv.drop(columns=['Checksum'], inplace=True)
      return df_pgv

    def save_to_csv(self, path = './'):
      self.train.to_csv(os.path.join(path, f'{self.name}.csv'), index=False)



class FitsilisDt:
    def __init__(self, datasets, id_=722):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.paper_id==self.resource_id]
      self.name = 'fitsilis'
      self.repo_url = self.resource.iloc[0].URL
      self.splits = {'train'}
      self.train = self.download()


    def get(self, split='train'):
      assert split in {'train'}
      return self.train

    def _read_file_content(self, filename):
        try:
            with open(os.path.join(str(self.resource_id), "Parliamentary Questions Corpus", f"{filename}.txt"), 'r', encoding='utf-16') as file:
                content = file.read()
                content_utf8 = content.encode('utf-8')
                return content_utf8
        except FileNotFoundError:
            return None

    def download(self):
      zenodo_download(self.resource_id, self.repo_url)
      with zipfile.ZipFile(f"{self.resource_id}/Parliamentary Questions Corpus.zip", 'r') as zip_ref:
        zip_ref.extractall(str(self.resource_id))

      df_parl_quest = pd.read_csv(f"{self.resource_id}/Parliamentary Questions Corpus Metadata.csv", delimiter=";")
      # Apply the function to replace the 'link serialNr' column with the file contents
      df_parl_quest['text'] = df_parl_quest['link serialNr'].apply(self._read_file_content)
      df_parl_quest['text'] = df_parl_quest['text'].str.decode('utf-8')
      return df_parl_quest

    def save_to_csv(self, path = './'):
      self.train.to_csv(os.path.join(path, f'{self.name}.csv'), index=False)


class BarziokasDt:
    def __init__(self, datasets, id_=285):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.paper_id==self.resource_id]
      self.name = 'barsiokas'
      self.repo_url = self.resource.iloc[0].URL
      self.down_folder = 'dataset'  # Data folder path within the git repository
      self.branch = "master"
      self.splits = {'train'}
      self.word_based_data = self.download()
      self.train = self.reduce()

    def get(self, split='train'):
      assert split in {'train'}
      return self.train

    def reduce(self):
      sentences,gt4,gt18 = {},{},{}
      counter = 0
      for index, row in self.word_based_data.iterrows():
        if len(str(row['Sentence#'])) > 5:
          counter += 1
          sentences[counter] = [row['Word']]
          gt4[counter] = [row['NE_4Tagset']]
          gt18[counter] = [row['NE_18Tagset']]
        else:
          sentences[counter].append(row['Word'])
          gt4[counter].append(row['NE_4Tagset'])
          gt18[counter].append(row['NE_18Tagset'])
      return pd.DataFrame({'sentence':sentences, 'tags4': gt4, 'tags18':gt18})


    def download(self):
      git_sparse_checkout_download(self.resource_id, self.repo_url, self.down_folder, self.branch)
      barziokas_4_df = pd.read_csv(f"{self.resource_id}/elNER4/elNER4_iobes.csv")
      barziokas_4_df = barziokas_4_df.rename(columns={'Tag': 'NE_4Tagset'})
      barziokas_18_df = pd.read_csv(f"{self.resource_id}/elNER18/elNER18_iobes.csv")
      barziokas_18_df = barziokas_18_df.rename(columns={'Tag': 'NE_18Tagset'})

      barziokas_df = pd.merge(barziokas_4_df, barziokas_18_df, left_index=True, right_index=True, how='inner')
      barziokas_df = barziokas_df.drop(['Sentence #_y', 'Word_y', 'POS_y'], axis=1)
      barziokas_df.columns = ['Sentence#', 'Word', 'POS', 'NE_4Tagset', 'NE_18Tagset']
      return barziokas_df

    def save_to_csv(self, path = './'):
      self.train.to_csv(os.path.join(path, f'{self.name}.csv'), index=False)


class PapaloukasDt:
    def __init__(self, datasets, id_=250):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.paper_id==self.resource_id]
      self.name = 'papaloukas'
      self.dataset_name = 'AI-team-UoA/greek_legal_code'
      self.subsets = ["volume", "chapter", "subject"]
      self.splits = {"train", "validation", "test"}
      self.dataset = self.download()

    def download(self, csv_datasets_folder='./'):
      df_dict = huggingface_download(self.resource_id, csv_datasets_folder, self.dataset_name, self.splits, subsets=self.subsets)

      df_splits = {}
      for split in self.splits:
        df_split_list = [df_ for name, df_ in df_dict.items() if split in name]
        df_split = reduce(lambda  left,right: pd.merge(left,right,left_index=True, right_index=True, how='inner'), df_split_list)
        df_split = df_split.drop(['text_x', 'text_y'], axis=1)
        df_splits[split] = df_split

      return df_splits

    def get(self, split = 'train'):
      assert split in self.splits
      return self.dataset[split]

    def save_to_csv(self, path = './'):
      self.train.to_csv(os.path.join(path, f'{self.name}.csv'), index=False)



class ProkopidisCrawledDt:
    def __init__(self, datasets, id_=284):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.paper_id==self.resource_id]
      self.name = 'prokopidis'
      self.repo_url = self.resource.iloc[0].URL
      self.splits = {'train'}
      self.train = self.download()

    def get(self, split='train'):
      assert split in self.splits
      return self.train


    def download(self):
      wget_download(self.resource_id, f"{self.repo_url}/resources/greek_corpus.tar.gz")
      tar_file_path = f"{self.resource_id}/greek_corpus.tar.gz"

      with tarfile.open(tar_file_path, "r:gz") as tar:
        # Extract all files to the current directory
        tar.extractall()

      data_folder = 'data-20130219-20191231'
      data = []
      for filename in os.listdir(data_folder):
        with open(os.path.join(data_folder, filename), 'r') as f:
          file_content = f.read()
          data.append({"text": file_content, "filename": filename.split(".txt")[0]})

      df = pd.DataFrame(data)
      return df

    def save_to_csv(self, path = './'):
      self.train.to_csv(os.path.join(path, f'{self.name}.csv'), index=False)


class DritsaDt:
    def __init__(self, datasets, id_=728):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.paper_id==self.resource_id]
      self.name = 'dritsa'
      self.repo_url = self.resource.iloc[0].URL.split(",")[0]
      self.splits = {'train'}
      self.train = self.download()
      self.train['text'] = self.train['speech']

    def get(self, split='train'):
      assert split in self.splits
      return self.train

    def download(self):
      zenodo_download(self.resource_id, self.repo_url)
      target_file = 'dataset_versions/tell_all.csv'
      with zipfile.ZipFile(f"{self.resource_id}/Greek Parliament Proceedings Dataset_Support Files_Word Usage Change Computations.zip", 'r') as zip_ref:
          for member in zip_ref.namelist():
            if member == target_file:
              csv_path = zip_ref.extract(member)
              df_728 = pd.read_csv(csv_path)
              return df_728

    def save_to_csv(self, path = './'):
      self.train.to_csv(os.path.join(path, f'{self.name}.csv'), index=False)


class AntonakakiDt:
    def __init__(self, datasets, figshare_access_token, id_=428):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.paper_id==self.resource_id]
      self.name = 'antonakaki'
      self.repo_url = self.resource.iloc[0].URL
      self.splits = {'train'}
      self.train = self.download()
      self.ACCESS_TOKEN = figshare_access_token
      self.BASE_URL = 'https://api.figshare.com/v2'
      self.ARTICLE_ID = '5492443'
        
    
    def get_article_details(self):
        url = f"{self.BASE_URL}/articles/{self.ARTICLE_ID}"
        headers = {
            'Authorization': f'token {self.ACCESS_TOKEN}'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    
    def download_file(self, file_info):
        file_url = file_info['download_url']
        file_name = file_info['name']
        print(f"Downloading {file_name} from {file_url}")
        
        response = requests.get(file_url)
        response.raise_for_status()
        
        with open(file_name, 'wb') as file:
            file.write(response.content)
        print(f"{file_name} has been downloaded successfully.")
    
    def get(self, split='train'):
      assert split in self.splits
      return self.train

    def download(self):
      article_details = self.get_article_details(self.ARTICLE_ID)
      
      for file_info in article_details.get('files', []):
        if file_info['name'] in ['ht_common_final_greek_sorted_reversed_with_SENTIMENT_20160419.txt', 'ht_sorted_unique_with_SENTIMENT_20160419.txt']:
          self.download_file(file_info)

      data_types = {'tweet_id': str, 'text': str}
      ref_df = pd.read_csv('428_referendum_sentiment.csv', dtype=data_types)
      ref_df.drop(columns=["positive", "negative", "sentiment"], inplace=True)
      ref_df.drop_duplicates(subset=['text'], inplace=True)

      elect_df = pd.read_csv('428_elections_sentiment.csv', dtype=data_types)
      elect_df.drop(columns=["positive", "negative", "sentiment"], inplace=True)
      elect_df.drop_duplicates(subset=['text'], inplace=True)

      df_428 = pd.concat([elect_df, ref_df], axis=0)
      df_428.dropna(inplace=True)
      return df_428

    def save_to_csv(self, path = './'):
      self.train.to_csv(os.path.join(path, f'{self.name}.csv'), index=False)

