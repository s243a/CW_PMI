import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
import json
import os
from google.colab import drive
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from io import BytesIO
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from tqdm import tqdm
import math
from scipy import stats
import csv
from itertools import chain
from enum import Enum
from collections import Counter

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class CombinedWordFreq:
    def __init__(self, unigram_counts=None, ngram_counts=None, container=None, lemmatized_mapping=None):
        self.container = container
        self._unigram_counts = unigram_counts or Counter()
        self._ngram_counts = ngram_counts or Counter()
        self._lemmatized_mapping = lemmatized_mapping or {}
        self._lemmatizer = WordNetLemmatizer()

    @property
    def unigram_counts(self):
        if self.container is not None:
            if hasattr(self.container, 'unigram_counts') and not (self.container.unigram_counts is None):
                return self.container.unigram_counts
        return self._unigram_counts

    @property
    def ngram_counts(self):
        if self.container is not None:
            if hasattr(self.container, 'ngram_counts') and not (self.container.ngram_counts is None):
                return self.container.ngram_counts
        return self._ngram_counts

    def _lemmatize_key(self, key):
        return ' '.join(self._lemmatizer.lemmatize(word) for word in key.split())

    def _get_from_lemmatized(self, key):
        lemma = self._lemmatize_key(key)
        if lemma in self._lemmatized_mapping:
            return self._lemmatized_mapping[lemma].get(key, {})
        return {}

    def get_lemmatized_counts(self, key):
        data = self._get_from_lemmatized(key)
        if data:
            return tuple(data)  # Convert list to tuple
        return (0, 0)  # Default to (0, 0) if no data

    def get_lemmatized_doc_count(self, key):
        return self.get_lemmatized_counts(key)[0]

    def get_lemmatized_total_count(self, key):
        return self.get_lemmatized_counts(key)[1]
    def count(self):
        total = len(self.unigram_counts) + len(self.ngram_counts)
        for lemma in self._lemmatized_mapping:
            for key, value in self._lemmatized_mapping[lemma].items():
                total += value[1]
        return total
    def __getitem__(self, key):
        if self.unigram_counts is not None and key in self.unigram_counts:
            return self.unigram_counts[key]
        elif self.ngram_counts is not None and key in self.ngram_counts:
            return self.ngram_counts[key]
        else:
            key2=self._lemmatize_key(key)
            lemmas = self._get_from_lemmatized(key2)
            if key in lemmas:
              return lemmas[key][1] # Return the total count (second element of the list)
            raise KeyError(key)
    def lemma_contains_form(self,key):
        if len(self._lemmatized_mapping) == 0:
            return False
        key2=self._lemmatize_key(key)
        if not (key2 in self._lemmatized_mapping):
            return False
        return key in self._lemmatized_mapping[key2]
    def __contains__(self, key):
        return (self.unigram_counts is not None and key in self.unigram_counts) or \
               (self.ngram_counts is not None and key in self.ngram_counts) or \
               (self.lemma_contains_form(key))

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return iter(self)

    def values(self):
        return (self[key] for key in self)

    def items(self):
        return chain(
            self.unigram_counts.items(),
            self.ngram_counts.items(),
            ((key, value[1]) for lemma in self._lemmatized_mapping for key, value in self._lemmatized_mapping[lemma].items())
        )

    def __len__(self):
        return sum(1 for _ in self)

    def __iter__(self):
        return chain(
            self.unigram_counts.keys(),
            self.ngram_counts.keys(),
            (key for lemma in self._lemmatized_mapping for key in self._lemmatized_mapping[lemma])
        )

    def __repr__(self):
        return f"CombinedWordFreq(unigrams: {len(self.unigram_counts)}, ngrams: {len(self.ngram_counts)}, lemmatized: {sum(len(forms) for forms in self._lemmatized_mapping.values())})"
    def to_dict(self):
        return {
            'unigram_counts': dict(self._unigram_counts),
            'ngram_counts': dict(self._ngram_counts),
            'lemmatized_mapping': self._lemmatized_mapping
        }

    @classmethod
    def from_dict(cls, data, container=None):
        return cls(
            unigram_counts=data.get('unigram_counts'),
            ngram_counts=data.get('ngram_counts'),
            lemmatized_mapping=data.get('mega_mapping'),
            container=container
        )

class DF_Method(Enum):
    """
    Enumeration for different degrees of freedom calculation methods.

    Methods:
        ENTROPY: Uses Shannon entropy-based calculation with expected occurrences
        MIN: Uses the original minimum threshold approach
    """
    ENTROPY = "entropy"
    MIN = "min"

class EnhancedEfficientTfidfVectorizer:
    def __init__(self, max_features=10000, ngram_range=(1, 3), min_df=2, max_df=0.95,
                 use_idf=True, use_surprise=True, pmi_threshold=0,
                 store_results=True, use_google_drive=False, df_method=DF_Method.ENTROPY, ignore_saved=False):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.use_surprise = use_surprise
        self.pmi_threshold = pmi_threshold
        self.store_results = store_results
        self.use_google_drive = use_google_drive
        self.lemmatizer = WordNetLemmatizer()
        self.mega_mapping = None
        self.drive_service = None
        self.word_freq = CombinedWordFreq(container=self)
        #self.total_words = 0
        #self.unigram_counter = Counter()
        self._total_unigrams = 0
        self.total_words = 0
        self.document_count = 0
        self.df_method = df_method
        self.documents = None  # Add this
        self._document_length = 0  # Add this
        self.ignore_saved = ignore_saved

        if use_google_drive:
            self._initialize_google_drive()

    @property
    def unigram_counts(self):
        #if not self.word_freq._unigram_counts: #Same as self.word_freq._unigram_counts == 0?
        #    self.word_freq._unigram_counts = dict(self.unigram_counter)
        return self.word_freq._unigram_counts
    @property
    def total_unigrams(self):
        if self._total_unigrams == 0:
            #self._total_unigrams = sum(self.unigram_counter.values())
            self._total_unigrams = sum(self.unigram_counts.values())
        return self._total_unigrams

    def _count_unigrams(self, document):
        return Counter(word_tokenize(document.lower()))

    def _initialize_google_drive(self):
        auth.authenticate_user()
        self.drive_service = build('drive', 'v3')
        print("Authenticated with Google Drive.")

    def _calculate_pmi_with_t_score(self, ngram):

        # List of n-grams to debug
        debug_ngrams = ["jet propulsion", "propulsion laboratory", "do n't"]

        debug_mode = ngram in debug_ngrams

        words = ngram.split()
        if len(words) == 1:
            return 0

        # Calculate n-gram probability
        ngram_count = self.word_freq.get(ngram, 0)
        ngram_prob = ngram_count / self.total_unigrams

        if debug_mode:
            print(f"\nAnalyzing '{ngram}':")
            print(f"ngram count: {ngram_count}")
            print(f"total unigrams: {self._total_unigrams}")
            print(f"ngram probability: {ngram_prob}")

            # Get individual word stats
            for word in words:
                count = self.word_freq.get(word, 0)
                prob = count / self._total_unigrams
                print(f"'{word}' count: {count}, prob: {prob}")

            # Print expected vs actual probability
            expected_prob = np.prod([self.word_freq.get(word, 0) / self._total_unigrams for word in words])
            print(f"Expected probability: {expected_prob}")
            print(f"Actual vs Expected ratio: {ngram_prob/expected_prob}")
            print(f"log of ratio (before smoothing): {math.log(ngram_prob/expected_prob)}")

        #individual_counts = [self.word_freq.get(word, 0) for word in words]
        #individual_probs = [count / self._total_unigrams for count in individual_counts]

        # Calculate n-gram standard error
        #ngram_df = max(self.total_unigrams * ngram_prob, 5) #ToDo: Review this min value of 5
        #ngram_se = math.sqrt((ngram_prob * (1-ngram_prob)) / ngram_df)

        # Calculate individual word probabilities and errors
        word_stats = []
        for word in words:
            count = self.word_freq.get(word, 0)
            prob = count / self.total_unigrams
            df = max(self.total_unigrams * prob, 5)
            se = math.sqrt((prob * (1-prob)) / df)

            word_stats.append({
                'prob': prob,
                'se': se,
                'df': df
            })

        #expected_prob = np.prod(individual_probs)

        # Calculate expected probability and error
        expected_prob = np.prod([ws['prob'] for ws in word_stats])
        #relative_variance = sum((ws['se']/ws['prob'])**2 for ws in word_stats)
        rv_num = sum((ws['se']/ws['prob'])**2 for ws in word_stats)
        rv_den = sum((1/ws['prob'])**2 for ws in word_stats)
        relative_variance = rv_num / rv_den
        # Calculate smoothing parameters

        if self.df_method == DF_Method.ENTROPY:
            ngram_df=self.calculate_entropy_df(ngram,debug_mode=debug_mode)
        elif self.df_method == DF_Method.MIN:
            ngram_df=min(ws['df'] for ws in word_stats)
        else:
            raise Exception('Uknown degrees of freedom method')
        if ngram_df > 4:
            ngram_se = math.sqrt((ngram_prob * (1-ngram_prob)) / ngram_df)
        else:
            ngram_se = 0
        # Add check for zero standard error
        if ngram_se == 0:
            confidence = 0  # Or some other appropriate default
            t_stat = 1
            p_value = 1
            w = 1
        else:
            t_stat = abs((ngram_prob - expected_prob)/ngram_se)
            p_value = 2 * (1 - stats.t.cdf(t_stat, df=ngram_df))  # Two-tailed test
            w = 1/p_value  # Or some function of p_value that grows with surprise
            confidence = 1 - stats.t.cdf(t_stat, df=ngram_df)  # Use the calculated df

        #alpha = math.sqrt(relative_variance) / math.sqrt(self.total_unigrams)
        alpha = math.sqrt(relative_variance) / math.sqrt(ngram_df)

        # Apply smoothing to both probabilities
        #smoothed_ngram_prob = (ngram_count + alpha) / (self.total_unigrams + alpha * confidence)
        smoothed_ngram_prob = (w*ngram_count/self.total_unigrams + alpha *(1 - confidence))/(1+w)
        smoothed_expected_prob = (w*expected_prob + alpha * (1 - confidence))/(1+w)

        # Calculate PMI with smoothed probabilities
        pmi = math.log(max(smoothed_ngram_prob, 1e-10) / max(smoothed_expected_prob, 1e-10))

        if debug_mode:
            print("\nSmoothing calculations:")
            print(f"ngram_df: {ngram_df}")
            print(f"ngram_se: {ngram_se}")

            if self.df_method == DF_Method.ENTROPY:
                print("Using Entropy method for df")
            else:
                print("Using Min method for df")
            print(f"df used: {ngram_df}")  # Use the already calculated ngram_df

            print(f"\nRelative variance components:")
            for word, word_stat in zip(words, word_stats):
                print(f"'{word}': se/prob = {word_stat['se']/word_stat['prob']}")
            print(f"Total relative variance: {relative_variance}={rv_num}/{rv_den}")

            print(f"\nConfidence calculation:")

            if ngram_se>0:
                print(f"t-statistic: {ngram_prob/ngram_se}={ngram_prob}/{ngram_se}=ngram_prob/ngram_se")
            else:
                print(f"t-statistic: 0")
            print(f"confidence: {confidence}")
            print(f"alpha: {alpha}")
            print(f"smoothed_expected_prob={smoothed_expected_prob}")
            print(f"smoothed_ngram_prob={smoothed_ngram_prob}")
            print(f"pmi={pmi} for {ngram}")
        return pmi

    def calculate_entropy_df(self, ngram,debug_mode=False):
        words = ngram.split()

        if debug_mode:
            print(f"\nCalculating entropy df for '{ngram}':")

        # Get unigram probabilities
        unigram_probs = []
        for word in words:
            count = self.word_freq.get(word, 0)
            prob = count / self.total_unigrams
            unigram_probs.append(prob)

        # Calculate Shannon entropy from unigram probabilities
        H = -sum(p * math.log(p) for p in unigram_probs if p > 0)

        # Calculate expected occurrences using Bayesian approach
        doc_length = self._document_length
        ngram_length = len(words)
        possible_positions = doc_length - ngram_length + 1

        # Prior from unigrams
        prior_prob = np.prod(unigram_probs)

        # Actual observations
        observed_count = self.word_freq.get(ngram, 0)
        total_unigrams=self.total_unigrams
        # Calculate entropy of n-gram occurrence
        ngram_H = -sum([
            p * math.log(p)
            for p in [observed_count/total_unigrams, 1-observed_count/total_unigrams]
            if p > 0
        ])

        # Normalize beta to be between 0 and 1
        if H > 0:
            beta = math.exp(ngram_H) / max(math.exp(ngram_H), math.exp(H))
            alpha = 1 - beta
        else:
            alpha = 1  # If H=0, use only prior

        posterior_prob = (alpha * prior_prob + (1-alpha) * (observed_count / self.total_unigrams))
        expected_occurrences = posterior_prob * possible_positions

        # Final df calculation
        df = math.exp(H) * expected_occurrences - ngram_length

        if debug_mode:
            print(f"Unigram probabilities: {unigram_probs}")
            print(f"Entropy (H): {H}")
            print(f"exp(H): {math.exp(H)}")
            print(f"N-gram entropy (ngram_H): {ngram_H}")
            print(f"exp(ngram_H): {math.exp(ngram_H)}")
            print(f"Beta (observation weight): {beta}")  # Added beta output
            print(f"\nBayesian calculation:")
            print(f"Prior probability: {prior_prob}")
            print(f"Observed count: {observed_count}")
            print(f"Alpha (prior weight): {alpha}")
            print(f"Posterior probability: {posterior_prob}")
            print(f"\nFinal calculation:")
            print(f"Possible positions: {possible_positions}")
            print(f"Expected occurrences: {expected_occurrences}")
            print(f"N-gram length: {ngram_length}")
            print(f"Final df: {max(5, round(df))}")

            #print(f"doc_length={doc_length}")
            #print(f"ngram_prob={ngram_prob}")

        return max(5, round(df))

    def _map_function(self, document):
        words = nltk.word_tokenize(document.lower())
        ngrams = []

        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            ngrams.extend([' '.join(words[i:i+n]) for i in range(len(words) - n + 1)])

        selected_ngrams = [ngram for ngram in ngrams
                           if self._calculate_pmi_with_t_score(ngram) > self.pmi_threshold]

        return [(ngram, 1, 1) for ngram in selected_ngrams]


    def _reduce_function(self, ngram, values):
        doc_ids = set()
        total_count = 0
        for doc_id, count in values:
            doc_ids.add(doc_id)
            total_count += count
        lemma = ' '.join([self.lemmatizer.lemmatize(word) for word in ngram.split()])
        return (lemma, ngram, len(doc_ids), total_count)

    def _build_mega_mapping(self, raw_documents):
        # First pass: count word frequencies
        #self.word_freq = Counter()

        self.document_count = len(raw_documents)

        for doc in raw_documents:
            self.unigram_counts.update(self._count_unigrams(doc))

        # Ensure unigram_counts is updated
        _ = self.unigram_counts
        self._total_unigrams = sum(self.unigram_counts.values())


        # Filter unigrams based on document frequency
        min_count = max(self.min_df, 2) if isinstance(self.min_df, int) else int(self.min_df * self.document_count)
        max_count = min(self.max_df * self.document_count, self.document_count - 1) if isinstance(self.max_df, float) else self.max_df

        valid_unigrams = {word for word, count in self.unigram_counts.items() if min_count <= count <= max_count}

        # Second pass: generate and evaluate higher-order n-grams
        self.ngram_counts = {word: count for word, count in self.unigram_counts.items() if word in valid_unigrams}

        for n in range(2, self.ngram_range[1] + 1):
            # First pass: collect all counts for this n-gram size
            new_ngrams = {}
            for doc in raw_documents:
                words = word_tokenize(doc.lower())
                for i in range(len(words) - n + 1):
                    ngram = ' '.join(words[i:i+n])
                    if all(word in valid_unigrams for word in words[i:i+n]):
                        new_ngrams[ngram] = new_ngrams.get(ngram, 0) + 1

            # Update ngram_counts with these new counts
            self.ngram_counts.update(new_ngrams)

            # For n > 2, filter based on PMI
            if n > 2:
                # Calculate PMI for all n-grams of this size
                filtered_ngrams = {
                    ngram: count for ngram, count in new_ngrams.items()
                    if self._calculate_pmi_with_t_score(ngram) > self.pmi_threshold
                }
                # Update with only the filtered n-grams
                self.ngram_counts = {
                    k: v for k, v in self.ngram_counts.items()
                    if ' ' not in k or len(k.split()) < n or k in filtered_ngrams
                }

        # Consider a map reduce impelmentation
        # # Second pass: build mega mapping
        # with Pool() as pool:
        #    mapped_data = pool.map(self._map_function, raw_documents)
        #
        # ngram_data = defaultdict(list)
        # for doc_id, doc_data in enumerate(mapped_data):
        #    for ngram, _, count in doc_data:
        #        ngram_data[ngram].append((doc_id, count))

        mega_mapping = {}

        # Include unigrams
        for unigram, count in self.unigram_counts.items():
            lemma = self.lemmatizer.lemmatize(unigram)
            if lemma not in mega_mapping:
                mega_mapping[lemma] = {}
            mega_mapping[lemma][unigram] = (count, 0)  # PMI for unigrams is 0 by definition

        # Include n-grams
        for ngram, count in self.ngram_counts.items():
            if ' ' in ngram:  # Ensure we're only processing n-grams (n > 1)
                lemma = ' '.join([self.lemmatizer.lemmatize(word) for word in ngram.split()])
                if lemma not in mega_mapping:
                    mega_mapping[lemma] = {}
                mega_mapping[lemma][ngram] = (count, self._calculate_pmi_with_t_score(ngram))

        #for ngram, values in tqdm(ngram_data.items(), desc="Building mega mapping"):
        #    lemma, form, doc_freq, total_count = self._reduce_function(ngram, values)
        #    if lemma not in mega_mapping:
        #        mega_mapping[lemma] = {}
        #    mega_mapping[lemma][form] = (doc_freq, total_count)

        return mega_mapping

    def _load_or_build_mega_mapping(self, raw_documents):
        if self.store_results and not self.ignore_saved:  # Check ignore_saved
            data = self._load_from_drive('mega_mapping.json') if self.use_google_drive else self._load_from_file('mega_mapping.json')

            if data:
                print("Loaded mega_mapping from storage.")

                if 'mega_mapping' in data:
                    # Structured data
                    self.word_freq = CombinedWordFreq.from_dict(data, container=self)
                    self.ensure_total_unigrams(raw_documents)
                    return data['mega_mapping']
                elif isinstance(data, dict):
                    # Flat data structure
                    first_value = next(iter(data.values()))
                    if isinstance(first_value, dict):
                        # Lemmatized mapping
                        self.word_freq = CombinedWordFreq(lemmatized_mapping=data, container=self)
                    else:
                        # Ngram counts
                        self.word_freq = CombinedWordFreq(ngram_counts=data, container=self)

                    self.ensure_unigram_counts(raw_documents)
                    return data
                else:
                    print("Unexpected data format in stored file.")
                    return self._build_mega_mapping(raw_documents)

        print("Building new mega_mapping...")
        mega_mapping = self._build_mega_mapping(raw_documents)

        if self.store_results:
            data_to_store = {
                'mega_mapping': mega_mapping,
                'unigram_counts': dict(self.unigram_counts),
                'ngram_counts': dict(self.word_freq.ngram_counts),
                'total_unigrams': self._total_unigrams
            }
            if self.use_google_drive:
                self._save_to_drive('mega_mapping.json', data_to_store)
            else:
                self._save_to_file('mega_mapping.json', data_to_store)

        return mega_mapping

    def _load_from_drive(self, filename):
        try:
            results = self.drive_service.files().list(q=f"name='{filename}'", spaces='drive', fields='files(id)').execute()
            items = results.get('files', [])
            if items:
                file_id = items[0]['id']
                request = self.drive_service.files().get_media(fileId=file_id)
                fh = BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                fh.seek(0)
                return json.loads(fh.read().decode())
        except Exception as e:
            print(f"Error loading from Google Drive: {str(e)}")
        return None

    def _save_to_drive(self, filename, data):
        try:
            file_metadata = {'name': filename}
            media = MediaIoBaseUpload(BytesIO(json.dumps(data).encode()), mimetype='application/json')
            self.drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            print(f"Saved {filename} to Google Drive.")
        except Exception as e:
            print(f"Error saving to Google Drive: {str(e)}")

    def _load_from_file(self, filename):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return None

    def _save_to_file(self, filename, data):
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Saved {filename} locally.")

    def fit(self, raw_documents):
        self.documents = raw_documents  # Store documents
        # Calculate average document length
        self._document_length = sum(len(doc.split()) for doc in raw_documents)
        self.mega_mapping = self._load_or_build_mega_mapping(raw_documents)
        return self

    def ensure_total_unigrams(self,raw_documents):
      if self._total_unigrams == 0:
          if self.word_freq.unigram_counts:
              self._total_unigrams = sum(self.word_freq.unigram_counts.values())
          else:
              self._initialize_unigram_freq(raw_documents)
    def ensure_unigram_counts(self, raw_documents=None):
        if not self.unigram_counts and raw_documents:
            self._initialize_unigram_freq(raw_documents)
 #   def _initialize_unigram_freq(self, raw_documents):
    def _initialize_unigram_freq(self, raw_documents):
        self.word_freq._unigram_counts = Counter()
        for doc in raw_documents:
            words = nltk.word_tokenize(doc.lower())
            self.word_freq._unigram_counts.update(words)
        self._total_unigrams = sum(self.word_freq._unigram_counts.values())

    def transform(self, raw_documents):
        # TODO: Implement document transformation using mega_mapping
        pass

    def fit_transform(self, raw_documents):
        return self.fit(raw_documents).transform(raw_documents)

    def save_ngram_stats_to_csv(self, filename='ngram_stats.csv'):
        drive_path = '/content/drive/My Drive/'
        full_path = drive_path + filename
        with open(full_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['n-gram', 'PMI', 'Frequency', 'Document Frequency', 'Individual Word Probabilities']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for ngram in self.word_freq:
                if ' ' in ngram:  # Only consider n-grams (not unigrams)
                    pmi = self._calculate_pmi_with_t_score(ngram)
                    freq = self.word_freq[ngram]
                    doc_freq = self.mega_mapping.get(ngram, {}).get(ngram, (0, 0))[0]
                    individual_probs = [self.word_freq.get(word, 0) / self._total_unigrams for word in ngram.split()]

                    writer.writerow({
                        'n-gram': ngram,
                        'PMI': pmi,
                        'Frequency': freq,
                        'Document Frequency': doc_freq,
                        'Individual Word Probabilities': ' '.join(map(str, individual_probs))
                    })

        print(f"Saved n-gram statistics to {filename}")
    def __iter__(self):
        return iter(self.word_freq)

    def items(self):
        return self.word_freq.items()

    def keys(self):
        return self.word_freq.keys()

    def values(self):
        return self.word_freq.values()

    def __len__(self):
        return len(self.word_freq)

    def __getitem__(self, key):
        return self.word_freq[key]

    def get(self, key, default=None):
        return self.word_freq.get(key, default)

    def __contains__(self, key):
        return key in self.word_freq
# Add this method to your EnhancedEfficientTfidfVectorizer class

# Usage example
from sklearn.datasets import fetch_20newsgroups

categories = ['sci.space']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))

vectorizer = \
  EnhancedEfficientTfidfVectorizer(
    max_features=10000, ngram_range=(1, 3),
    pmi_threshold=0, store_results=True,
    use_google_drive=True, ignore_saved=True)
vectorizer.fit(twenty_train.data)

# Print some information about the mega_mapping
print("Number of lemmas:", len(vectorizer.mega_mapping))
sample_lemma = next(iter(vectorizer.mega_mapping))
print(f"Sample lemma '{sample_lemma}' forms:")
for form, (doc_freq, total_count) in vectorizer.mega_mapping[sample_lemma].items():
    print(f"  Form: '{form}', Document Frequency: {doc_freq}, Total Count: {total_count}")

# Print some high-PMI n-grams
print("\nSome high-PMI n-grams:")
if vectorizer.word_freq is not None:
    high_pmi_ngrams = sorted(
        [(ngram, vectorizer._calculate_pmi_with_t_score(ngram)) for ngram in vectorizer.word_freq if ' ' in ngram],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for ngram, pmi in high_pmi_ngrams:
        print(f"  '{ngram}': PMI = {pmi:.2f}")
else:
    print("Error: word_freq is None. Unable to calculate PMI scores.")