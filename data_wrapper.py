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
# !pip install conll-df
from conll_df import conll_df
from sklearn.model_selection import train_test_split

def wget_download(resource_id, url):
  os.makedirs(str(resource_id), exist_ok=True)
  # Use wget to download the file (as in >> !wget -P {resource_id} {url})
  wget.download(url=url, out=resource_id)

import os
import subprocess
import shlex

def zenodo_download(output_dir: str, zenodo_url: str):
    """
    Downloads a Zenodo record using the zenodo-get command-line tool.

    Args:
        output_dir: The directory where the files will be saved.
        zenodo_url: The full Zenodo URL or the Zenodo Record ID/DOI.
    """
    # 1. Check if the directory already exists and contains content
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"Directory exists and is not empty: {output_dir}. Skipping download.")
        return

    # 2. Create the output directory if it doesn't exist
    print(f"Creating directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 3. Construct and run the command
    print(f"Starting download from {zenodo_url}...")
    
    # Construct the command as a string
    # The zenodo-get tool can take the full URL, the DOI, or just the Record ID
    command = f"zenodo_get -o {shlex.quote(output_dir)} {shlex.quote(zenodo_url)}"
    
    try:
        # Execute the command. check=True raises an error for non-zero exit codes.
        subprocess.run(
            shlex.split(command), 
            check=True, 
            capture_output=True, 
            text=True
        )
        print("✅ Download complete.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Zenodo download failed with error: {e}")
        print(f"Stderr: {e.stderr}")
        # Optionally, clean up the created directory if the download fails
        # os.rmdir(output_dir) 
    except FileNotFoundError:
        print("❌ Error: 'zenodo_get' command not found. Ensure the tool is installed (pip install zenodo-get) and accessible in your environment's PATH.")


def huggingface_download(resource_id, dataset_name, splits, subsets=[None]):
  """
  Download the data from HuggingFace
  """
  df_dict = {}
  for subset in subsets:
    if dataset_name == "strombergnlp/offenseval_2020" and subset == "gr":
            # Manual URL mapping for Greek subset
            data_files = {
                "train": "https://huggingface.co/datasets/strombergnlp/offenseval_2020/resolve/main/offenseval-gr-training-v1.tsv",
                "test": "https://huggingface.co/datasets/strombergnlp/offenseval_2020/resolve/main/offenseval-gr-test-v1.tsv",
            }

            dataset = load_dataset(
                "csv",
                data_files=data_files,
                sep="\t"
            )
    else:
      dataset = load_dataset(dataset_name, subset)

    for split in splits:
      df_hg = pd.DataFrame(dataset[split])
      if resource_id == 250: # The Papaloukas dataset
        df_hg = df_hg.rename(columns={'label': subset})

      if len(subsets) > 1:
        df_dict[f"{split}_{subset}"] = df_hg
      else:
        df_dict[f"{split}"] = df_hg
  return df_dict

def run_git_command(command, cwd=None):
    result = subprocess.run(command, shell=True, cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip()

def git_sparse_checkout_download(resource_id, repo_url, to_download, branch, root_dir):
  # Move to the root directory
  os.chdir(root_dir)
  repo_dir = os.path.join(root_dir, f'repo_{resource_id}')
  if os.path.exists(repo_dir):
    print(f"Items exists in directory: {repo_dir}")
    return

  # Initialize the git repository
  run_git_command(f'git init repo_{resource_id}')
  os.chdir(repo_dir)
  print(f"Download github items in directory: {repo_dir}")

  run_git_command(f'git remote add -f origin {repo_url}')
  run_git_command(f'git config core.sparseCheckout true')

  # Define the files to download by adding each file path to the sparse-checkout file
  with open('.git/info/sparse-checkout', 'w') as f:
      for item in to_download:
          f.write(item + '\n')

  # Pull the specific files from the repository
  run_git_command(f'git pull origin {branch}')

  # Verify if the files have been downloaded
  missing_items = []
  for item in to_download:
      if os.path.exists(item):
          print(f"Successfully downloaded {item}")
      else:
          missing_items.append(item)

  if missing_items:
      print(f"Failed to download: {', '.join(missing_items)}. Please check the paths and branch name.")

  # Move back to the root directory
  os.chdir(root_dir)


class BarzokasDt:
    def __init__(self, datasets, root_dir=os.getcwd(), id_=56):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.id==self.resource_id]
      self.repo_url = self.resource.iloc[0].url
      self.root_dir = root_dir
      self.down_items = ['data/corpora']  # Data folder path within the git repository
      self.branch = "master"
      self.name = "barzokas"
      self.splits = {'train'}
      self.train = self.download()

    def download(self):
      git_sparse_checkout_download(self.resource_id, self.repo_url, self.down_items, self.branch, self.root_dir)
      data_root_dir = os.path.join(self.root_dir, f'repo_{self.resource_id}', self.down_items[0])
      df_list = []
      for data_folder in os.listdir(data_root_dir):
        df_tsv = pd.read_csv(os.path.join(data_root_dir, data_folder, "metadata.tsv"), sep='\t')
        data = []
        txt_root_dir = os.path.join(data_root_dir, data_folder, "text")
        for txt_folder in os.listdir(txt_root_dir):
          for txt_file in os.listdir(os.path.join(txt_root_dir, txt_folder)):
              if txt_file.endswith('.txt'):
                  # Extract the ID from each TXT filename
                  txt_id = os.path.splitext(txt_file)[0]
                  # Read the content of each TXT file
                  with open(os.path.join(txt_root_dir, txt_folder, txt_file), 'r', encoding='utf-8') as f:
                      txt_content = f.read().replace('\n', '')
                  # Merge the ID, title, and text content into a new DataFrame
                  row = df_tsv[df_tsv['id'] == txt_id]
                  if not row.empty:
                      row_data = row.iloc[0].to_dict()
                      row_data['text'] = txt_content
                      row_data['status'] = txt_folder
                      data.append(row_data)

          # Create the final DataFrame
          df_publisher = pd.DataFrame(data)
          df_publisher["publisher"] = data_folder
          df_list.append(df_publisher)

      df = pd.concat(df_list)
      # remove repo dir
      shutil.rmtree(os.path.join(self.root_dir, f'repo_{self.resource_id}'))
      return df

    def get(self, split='train'):
      assert split in {'train'}
      return self.train

    def save_to_csv(self, path = './'):
      self.train.to_csv(os.path.join(path, f'{self.name}.csv'), index=False)


class KorreDt:
  def __init__(self, datasets, root_dir=os.getcwd(), id_=244):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.id==self.resource_id]
      self.name = 'korre'
      # Download data
      self.root_dir = root_dir
      self.repo_url = self.resource.iloc[0].url
      self.down_items = ['GNC']  # Data folder path within the git repository
      self.branch = "main"
      self.splits = {'train'}
      self.dataset = None
      self.train = self.download()

  def download(self):
      git_sparse_checkout_download(self.resource_id, self.repo_url, self.down_items, self.branch, self.root_dir)
      path = os.path.join(self.root_dir, f'repo_{self.resource_id}', self.down_items[0])
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
      self.resource = datasets.loc[datasets.id==self.resource_id]
      self.name = 'zampieri'
      # Download data
      self.repo_url = self.resource.iloc[0].url
      self.splits = ["train", "test"]
      self.dataset = self.download()
      self.train = self.dataset['train']
      self.test = self.dataset['test']

    def download(self):
      dataset_name = 'strombergnlp/offenseval_2020'
      subsets = ["gr"]
      df_dict = huggingface_download(self.resource_id, dataset_name, self.splits, subsets=subsets)
      return df_dict

    def get(self, split='train'):
      assert split in {'train', 'test'}
      return self.dataset[split]

    def save_to_csv(self, split='train', path = './'):
      assert split in {'train', 'test'}
      self.dataset[split].to_csv(os.path.join(path, f'{self.name}_{split}.csv'), index=False)


class ProkopidisMtDt:
    def __init__(self, datasets, id_=486):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.id==self.resource_id]
      self.repo_url = self.resource.iloc[0].url
      self.name = 'prokopidis_mt'
      self.langs_dict = {
          "eng": "English",
          "jpn": "Japanese",
          "fas": "Farsi"
      }
      self.target_langs = list(self.langs_dict.keys())
      self.target_lang_names = list(self.langs_dict.values())
      self.splits = {'train', 'test'}
      self.datasets = self.download()

    def _generate_checksum(self, text):
      return hashlib.sha256(text.encode()).hexdigest()

    def download(self):
      source_lang = "ell"
      repo_path = os.path.join(os.getcwd(), f"repo_{self.resource_id}")
      for other_lang in self.langs_dict:
        data_url = f"{self.repo_url}archives/ell-{other_lang}.zip"
        wget_download(repo_path, data_url)
        with zipfile.ZipFile(f"{repo_path}/ell-{other_lang}.zip", 'r') as zip_ref:
          zip_ref.extractall(f"{repo_path}/ell-{other_lang}")

      df_dict = dict()
      namespace = {'xml': 'http://www.w3.org/XML/1998/namespace'}
      # Iterate through TMX files in the directory
      for target_lang, target_langname in self.langs_dict.items():
        print(f'source: {source_lang}, target: {target_lang}')
        file_path = os.path.join(repo_path, f"ell-{target_lang}", "pgv",f"ell-{target_lang}.tmx")
        tree = ET.parse(file_path)
        root = tree.getroot()

        source_lang_text = []
        target_lang_text = []
        # Iterate through tu elements
        for tu in root.findall('.//tu'):
            source = tu.find(f'.//tuv[@xml:lang="{source_lang}"]/seg', namespaces=namespace).text
            target = tu.find(f'.//tuv[@xml:lang="{target_lang}"]/seg', namespaces=namespace).text
            source_lang_text.append(source)
            target_lang_text.append(target)

        df_pair = pd.DataFrame({'source': source_lang_text, 'target': target_lang_text})
        df_pair['original_index'] = df_pair.index
        df_pair['checksum'] = df_pair.source.apply(self._generate_checksum)
        df_grouped = df_pair.groupby('checksum', as_index=False).agg({
            'source': 'first',  # Keep the first occurrence of 'source' for each group
            'target': lambda x: list(set(x)),  # Convert 'target' values to a list of unique values
            'original_index': 'first'
        })
        # Sort by the original index to maintain the original order
        df_grouped.sort_values('original_index', inplace=True)
        df_grouped.drop(columns=['checksum', 'original_index'], inplace=True)
        df_grouped.reset_index(drop=True, inplace=True)
        
        test_df = df_grouped[df_grouped.index < 500]
        train_df = df_grouped[df_grouped.index >= 500]
        df_dict[target_lang] = {"train": train_df, "test": test_df}
        
      # Remove repo directory
      shutil.rmtree(repo_path)
      return df_dict

    def get(self, target_lang='eng', split='train'):
        assert target_lang in self.target_langs
        assert split in self.splits
        return self.datasets[target_lang][split]

    def save_to_csv(self, target_lang='eng', split='train', path = './'):
      assert target_lang in self.target_langs
      assert split in self.splits
      self.datasets[target_lang][split].to_csv(os.path.join(path, f'{self.name}_{target_lang}_{split}.csv'), index=False)


class BarziokasDt:
    def __init__(self, datasets, root_dir=os.getcwd(), id_=285):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.id==self.resource_id]
      self.name = 'barziokas'
      self.repo_url = self.resource.iloc[0].url
      self.root_dir = root_dir
      self.down_items = ['dataset']
      self.branch = "master"
      self.splits = {'train', 'test'}
      self.word_based_dataset = self.download()
      self.dataset = self.assemble_sentences()

    def assemble_sentences(self):
      '''
      Create full sentences from individual words
      '''
      sentences,gt4,gt18 = {},{},{}
      counter = 0
      for index, row in self.word_based_dataset.iterrows():
        if len(str(row['sentence'])) > 5:
          counter += 1
          sentences[counter] = [row['word']]
          gt4[counter] = [row['ne_tag4']]
          gt18[counter] = [row['ne_tag18']]
        else:
          sentences[counter].append(row['word'])
          gt4[counter].append(row['ne_tag4'])
          gt18[counter].append(row['ne_tag18'])
      
      df = pd.DataFrame({'sentence':sentences, 'ne_tag4': gt4, 'ne_tag18':gt18})
      df_dict = {
          'train': df[df.index >= 300],
          'test': df[df.index < 300]
      
      }
      return df_dict

    def download(self):
      git_sparse_checkout_download(self.resource_id, self.repo_url, self.down_items, self.branch, self.root_dir)
      dataset_path = os.path.join(self.root_dir, f'repo_{self.resource_id}', 'dataset')
      df_4tags = pd.read_csv(f"{dataset_path}/elNER4/elNER4_iobes.csv")
      df_4tags = df_4tags.rename(columns={'Tag': 'ne_tags4'})
      df_18tags = pd.read_csv(f"{dataset_path}/elNER18/elNER18_iobes.csv")
      df_18tags = df_18tags.rename(columns={'Tag': 'ne_tags18'})

      df_word = pd.merge(df_4tags, df_18tags, left_index=True, right_index=True, how='inner')
      df_word = df_word.drop(['Sentence #_y', 'Word_y', 'POS_y'], axis=1)
      df_word.columns = ['sentence', 'word', 'pos_tag', 'ne_tag4', 'ne_tag18']
      # Remove repo directory
      shutil.rmtree(os.path.join(self.root_dir, f'repo_{self.resource_id}'))
      return df_word

    def get(self, split='train'):
      assert split in self.splits
      return self.dataset[split]

    def save_to_csv(self, split='train', path = './'):
      assert split in self.splits
      self.dataset[split].to_csv(os.path.join(path, f'{self.name}.csv'), index=False)


class PapaloukasDt:
    def __init__(self, datasets, id_=250):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.id==self.resource_id]
      self.name = 'papaloukas'
      self.dataset_name = 'AI-team-UoA/greek_legal_code'
      self.subsets = ["volume", "chapter", "subject"]
      self.splits = {"train", "validation", "test"}
      self.dataset = self.download()

    def download(self):
      df_dict = huggingface_download(self.resource_id, self.dataset_name, self.splits, subsets=self.subsets)

      df_splits = {}
      for split in self.splits:
        df_split_list = [df_ for name, df_ in df_dict.items() if split in name]
        df_split = reduce(lambda  left,right: pd.merge(left,right,left_index=True, right_index=True, how='inner'), df_split_list)
        df_split = df_split.drop(['text_x', 'text_y'], axis=1)
        df_split = df_split[['text'] + self.subsets]
        df_splits[split] = df_split
      return df_splits

    def get(self, split = 'train'):
      assert split in self.splits
      return self.dataset[split]

    def save_to_csv(self, split='train', path = './'):
      assert split in self.splits
      self.dataset[split].to_csv(os.path.join(path, f'{self.name}_{split}.csv'), index=False)


class ProkopidisCrawledDt:
    def __init__(self, datasets, id_=284):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.id==self.resource_id]
      self.name = 'prokopidis_crawled'
      self.repo_url = self.resource.iloc[0].url
      self.splits = {'train'}
      self.train = self.download()

    def download(self):
      repo_path = os.path.join(os.getcwd(), f'repo_{self.resource_id}')
      wget_download(repo_path, f"{self.repo_url}/resources/greek_corpus.tar.gz")

      tar_file_path = os.path.join(repo_path, "greek_corpus.tar.gz")
      with tarfile.open(tar_file_path, "r:gz") as tar:
        tar.extractall(path=repo_path)

      data_dir = os.path.join(repo_path, 'data-20130219-20191231')
      data = []
      for filename in os.listdir(data_dir):
        with open(os.path.join(data_dir, filename), 'r') as f:
          file_content = f.read()
          data.append({"text": file_content, "filename": filename.split(".txt")[0]})

      df = pd.DataFrame(data)
      # remove repo dir
      shutil.rmtree(repo_path)
      return df
    
    def get(self, split='train'):
      assert split in self.splits
      return self.train

    def save_to_csv(self, path = './'):
      self.train.to_csv(os.path.join(path, f'{self.name}.csv'), index=False)


class DritsaDt:
    def __init__(self, datasets, id_=728):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.id==self.resource_id]
      self.name = 'dritsa'
      self.repo_url = self.resource.iloc[0].url
      self.splits = {'train'}
      self.train = self.download()

    def get(self, split='train'):
      assert split in self.splits
      return self.train

    def download(self):
      repo_path = os.path.join(os.getcwd(), f'repo_{self.resource_id}')
      zenodo_download(repo_path, self.repo_url)

      zip_file = os.path.join(repo_path, 'Greek Parliament Proceedings Dataset_Support Files_Word Usage Change Computations.zip')
      target_file = os.path.join('dataset_versions', 'tell_all.csv')
      with zipfile.ZipFile(zip_file, 'r') as zip_ref:
          for member in zip_ref.namelist():
            if member == target_file:
              csv_path = zip_ref.extract(member)
              df = pd.read_csv(csv_path)
              df.rename(columns={'speech': 'text'}, inplace=True)
              # reorder columns
              columns = ['text'] + [col for col in df.columns if col != 'text']
              df = df[columns]
              # remove repo dir
              shutil.rmtree(repo_path)
              return df

    def save_to_csv(self, path = './'):
      self.train.to_csv(os.path.join(path, f'{self.name}.csv'), index=False)


class KoniarisDt:
    def __init__(self, datasets, id_=780):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.id==self.resource_id]
      self.name = 'koniaris'
      # Download data
      self.repo_url = self.resource.iloc[0].url
      self.splits = ["train", "validation", "test"]
      self.dataset = self.download()
      self.train = self.dataset['train']

    def download(self):
      dataset_name = 'DominusTea/GreekLegalSum'
      hf_splits = ['train']
      df_dict = huggingface_download(self.resource_id, dataset_name, hf_splits)
      # split is given by the column subset
      # training set
      df_dict['train']['subset'] = df_dict['train']['subset'].astype(int)
      print(df_dict['train']['subset'].value_counts())
      # validation set
      df_dict['validation'] = df_dict['train'].loc[df_dict['train']['subset'] == 1]
      df_dict['validation'] = df_dict['validation'].drop(columns=['subset'])
      df_dict['validation'].reset_index(drop=True, inplace=True)

      # testing set
      df_dict['test'] = df_dict['train'].loc[df_dict['train']['subset'] == 2]
      df_dict['test'] = df_dict['test'].drop(columns=['subset'])
      df_dict['test'].reset_index(drop=True, inplace=True)

      # training set
      df_dict['train'] = df_dict['train'].loc[df_dict['train']['subset'] == 0]
      df_dict['train'] = df_dict['train'].drop(columns=['subset'])
      df_dict['train'].reset_index(drop=True, inplace=True)

      return df_dict

    def get(self, split='train'):
      assert split in self.splits
      return self.dataset[split]

    def save_to_csv(self, split='train', path = './'):
      assert split in self.splits
      self.dataset[split].to_csv(os.path.join(path, f'{self.name}_{split}.csv'), index=False)


class ProkopidisUdDt:
  def __init__(self, datasets, root_dir=os.getcwd(), id_=438):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.id==self.resource_id]
      self.name = 'prokopidis_ud'
      # Download data
      self.root_dir = root_dir
      self.repo_url = self.resource.iloc[0].url
      self.down_items = ['el_gdt-ud-train.conllu', 'el_gdt-ud-dev.conllu', 'el_gdt-ud-test.conllu']
      self.branch = "master"
      self.splits = {'train', 'validation', 'test'}
      self.dataset = self.download()

  def download(self):
      git_sparse_checkout_download(self.resource_id, self.repo_url, self.down_items, self.branch, self.root_dir)
      df_dict = dict()
      for split in self.splits:
        substr_filename_split = 'dev' if split=='validation' else split
        path = os.path.join(self.root_dir, f'repo_{self.resource_id}', f'el_gdt-ud-{substr_filename_split}.conllu')
        df = conll_df(path, file_index=False)
        df_dict[split] = df
      # remove git repository
      shutil.rmtree(os.path.join(self.root_dir, f'repo_{self.resource_id}'))

      return df_dict

  def get(self, split='train'):
    assert split in self.splits
    return self.dataset[split]

  def save_to_csv(self, split='train'):
    assert split in self.splits
    self.dataset[split].to_csv(os.path.join(self.root_dir, f'{self.name}.csv'), index=False)


class RizouDt:
    def __init__(self, datasets, id_=777):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.id==self.resource_id]
      self.repo_url = self.resource.iloc[0].url
      self.repo_name = f'repo_{self.resource_id}'
      self.name = 'rizou'
      self.splits = {'train', 'test'}
      self.dataset = self.download()

    def download(self):
      wget_download(self.repo_name, self.repo_url)

      # Unzip
      with zipfile.ZipFile(os.path.join(self.repo_name, 'uniway.zip'), 'r') as zip_ref:
        zip_ref.extractall(self.repo_name)
      
      gr_path = os.path.join(self.repo_name, 'uniway', 'GR')
      file_data = []
      files = os.listdir(gr_path)
      for file in files:
          with open(os.path.join(gr_path, file), 'r', encoding='utf-8') as f:
              lines = f.readlines()
              file_data.append([line.strip() for line in lines])

      # Create a dataframe where each column is data from one file
      columns_dict = {'corpus.txt': 'text', 'entities.txt': 'ne_tags', 'intents.txt': 'intent'}
      df = pd.DataFrame({columns_dict[file_]: data for file_, data in zip(files, file_data)})

      # Shuffle and split the dataset into training and testing sets stratified 
      # by the intent column
      target_column = 'intent'
      df_train, df_test = train_test_split(
          df, test_size=0.2, stratify=df[target_column], 
          shuffle=True, random_state=42
          )
      # Remove repository directory
      shutil.rmtree(self.repo_name)

      return {'train': df_train, 'test': df_test}

    def get(self, split='train'):
          assert split in self.splits
          return self.dataset[split]
          
    def save_to_csv(self, split='train', path = './'):
      assert split in self.splits
      self.dataset[split].to_csv(os.path.join(path, f'{self.name}_{split}.csv'), index=False)


class PapantoniouDt:
    def __init__(self, datasets, id_=756):
      self.resource_id = id_
      self.resource = datasets.loc[datasets.id==self.resource_id]
      self.name = 'papantoniou'
      self.repo_url = self.resource.iloc[0].url
      self.splits = {'train', 'validation', 'test'}
      self.dataset = self.download()

    def download(self):
      repo_path = os.path.join(os.getcwd(), f'repo_{self.resource_id}')
      zenodo_download(repo_path, self.repo_url)

      zip_file = os.path.join(repo_path, 'ner_nel_greek_dataset.zip')
      with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(repo_path)
      
      df_dict = dict()
      for split in self.splits:
        filename = 'validate.json' if split=='validation' else f'{split}.json'
        file_path = os.path.join(repo_path, 'ner_nel_greek_dataset', filename)
        with open(file_path, 'r', encoding='utf-8') as f:
          data = json.load(f)

        df_dict[split] = pd.DataFrame(data['items'])
      # remove repo dir
      shutil.rmtree(repo_path)
      return df_dict

    def get(self, split='train'):
      assert split in self.splits
      return self.dataset[split]

    def save_to_csv(self, split='train', path = './'):
      assert split in self.splits
      self.dataset[split].to_csv(os.path.join(path, f'{self.name}_{split}.csv'), index=False)
