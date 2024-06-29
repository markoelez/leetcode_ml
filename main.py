import re
import os
import glob
import pickle
import random
import hdbscan
import asyncio
import tiktoken
import numpy as np
import pandas as pd
from functools import cache
from tqdm import tqdm, trange
from openai import AsyncOpenAI
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


OPENAI_API_TOKEN = os.environ['OPENAI_API_TOKEN']
TEXT_EMBEDDING_MODEL = 'text-embedding-3-small'

def load_dataset() -> dict[str, str]:
  fnames = [os.path.basename(f) for f in glob.glob(os.path.join('ds', '*')) if os.path.isfile(f)]
  h = {}
  for fname in fnames:
    k = fname.rstrip('.cpp')
    with open(os.path.join('ds', fname), 'r') as f:
      t = f.read()
      if len(tokenize(t)) > 8092: continue
      h[k] = t
  return h

def save(obj, fn):
  with open(fn, 'wb') as f:
    pickle.dump(obj, f)

def load(fn):
  with open(fn, 'rb') as f:
    return pickle.load(f)

@cache
def tokenize(text: str) -> list[int]:
  return tiktoken.encoding_for_model(TEXT_EMBEDDING_MODEL).encode(text)

@cache
async def embed(text: str, hf: bool = True):
  if hf: return SentenceTransformer('all-MiniLM-L6-v2').encode(text)
  client = AsyncOpenAI(api_key=OPENAI_API_TOKEN)
  res = await client.embeddings.create(input=text, model=TEXT_EMBEDDING_MODEL)
  return np.array(res.data[0].embedding)

async def compute_embeddings(inputs: dict[str, str], fname: str = 'embeddings.pkl') -> pd.DataFrame:
  if os.path.exists(fname):
    df = load(fname)
  else:
    embeddings = []
    for k, v in tqdm(list(ds.items())):
      e = await embed(v)
      embeddings.append({'problem': k, 'text': v, 'embedding': e})
    df = pd.DataFrame(embeddings)
    save(df, fname)
  return df

def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> int:
  e1 = np.array(emb1).reshape(1, -1)
  e2 = np.array(emb2).reshape(1, -1)
  return cosine_similarity(e1, e2)[0][0]

def get_similar_questions(url: str, df: pd.DataFrame) -> pd.DataFrame:
  pname = re.search(r'problems/([^/]+)/', url).group(1)
  emb = df[df['problem'] == pname].iloc[0].to_dict()['embedding']
  df['similarity'] = df['embedding'].apply(lambda x: compute_similarity(x, emb))
  df = df[df['similarity'] >= 0.8]
  df = df.sort_values(by='similarity', ascending=False)
  return df[['problem', 'similarity']].reset_index(drop=True)

async def sample(messages: list[str], temperature: float = 1.0, max_len: int = 512) -> str:
  client = AsyncOpenAI(api_key=OPENAI_API_TOKEN)
  res = await client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=temperature,
    max_tokens=max_len)
  return res.choices[0].message.content.strip()

async def label(cluster: list[str], semaphore: asyncio.Semaphore) -> str:
  messages = [{'role': 'user', 'content': '''\
  You are an expert at identifying common patterns in competitive programming questions.
  Given the following list of questions that share the same pattern, identify the pattern.
  Your identified pattern should be less than 3 words.
  # Problems:
  {cluster}
  Return just the pattern with no other commentary.'''.format(cluster='\n- '.join(cluster))}]
  async with semaphore:
    return await sample(messages, temperature=0.5, max_len=512)

def cluster(df: pd.DataFrame) -> list[list[str]]:
  embeddings = np.stack(df['embedding'].values)
  problems = df['problem'].values

  clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
  labels = clusterer.fit_predict(embeddings)

  clusters = defaultdict(set)
  for label, problem in zip(labels, problems):
    clusters[label].add(problem)
  return [list(x) for x in clusters.values()]

async def run_clustering(df: pd.DataFrame) -> pd.DataFrame:
  semaphore = asyncio.Semaphore(5)
  raw_clusters = cluster(df)
  labels = await asyncio.gather(*[label(x, semaphore) for x in raw_clusters])
  return pd.DataFrame([{'label': l, 'problems': c} for c, l in zip(raw_clusters, labels)])

async def main():
  
  h = load_dataset()

  df = await compute_embeddings(inputs=h, fname='embeddings.pkl')

  clusters = await run_clustering(df)
  print(clusters.head())

  #url = 'https://leetcode.com/problems/two-sum/description'
  #print(get_similar_questions(url, df))

if __name__ == '__main__': asyncio.run(main())
