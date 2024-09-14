import os
import re
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# initialize stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# function to preprocess a document (tokenization, stopword removal, stemming)
def preprocess(document):
    # lowercasing
    document = document.lower()
    # remove punctuation and tokenize
    tokens = word_tokenize(re.sub(r'\W+', ' ', document))
    # remove stop words and perform stemming
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return processed_tokens

# soundex function for spelling matching
def soundex(word):
    word = word.lower()
    soundex_mapping = {
        'b': '1', 'f': '1', 'p': '1', 'v': '1',
        'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
        'd': '3', 't': '3',
        'l': '4',
        'm': '5', 'n': '5',
        'r': '6'
    }
    first_letter = word[0].upper()
    encoded = first_letter
    prev_digit = ''
    
    for char in word[1:]:
        if char in soundex_mapping:
            digit = soundex_mapping[char]
            if digit != prev_digit:
                encoded += digit
                prev_digit = digit

    encoded = encoded[:4].ljust(4, '0')
    return encoded

# function to read all .txt files from the corpus folder and return documents with filenames
def load_documents_from_folder(folder_path):
    documents = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                documents.append(file.read())
            filenames.append(filename.replace('.txt', ''))  # remove .txt extension from output
    return documents, filenames

# function to build the inverted index, biword index, and positional index
def build_indexes(documents):
    inverted_index = defaultdict(set)
    biword_index = defaultdict(set)
    positional_index = defaultdict(lambda: defaultdict(list))
    
    for doc_id, document in enumerate(documents):
        # preprocess for inverted and positional index (with stemming)
        tokens = preprocess(document)  # stemming for single terms
        
        # tokenize without stemming for biword indexing
        raw_tokens = word_tokenize(document.lower())  # no stemming
        
        for pos, token in enumerate(tokens):
            # inverted index and positional index
            inverted_index[token].add(doc_id)
            positional_index[token][doc_id].append(pos)
        
        # biword index from raw (unstemmed) tokens
        for pos in range(1, len(raw_tokens)):
            biword = raw_tokens[pos - 1] + " " + raw_tokens[pos]
            biword_index[biword].add(doc_id)
    
    return inverted_index, biword_index, positional_index


# function to perform boolean search query
def boolean_search(query, inverted_index, filenames):
    query = query.lower()
    terms = re.findall(r'\w+', query)
    
    result_set = set()
    operator = None
    
    for term in terms:
        # apply stemming
        term = stemmer.stem(term)
        
        if term == "and":
            operator = "AND"
        elif term == "or":
            operator = "OR"
        elif term == "not":
            operator = "NOT"
        else:
            if not result_set:
                result_set = inverted_index.get(term, set())
            else:
                if operator == "AND":
                    result_set = result_set.intersection(inverted_index.get(term, set()))
                elif operator == "OR":
                    result_set = result_set.union(inverted_index.get(term, set()))
                elif operator == "NOT":
                    result_set = result_set.difference(inverted_index.get(term, set()))
                    
            operator = None
    
    result_filenames = [filenames[doc_id] for doc_id in result_set]
    return result_filenames

# function to handle phrase queries using biword index
def phrase_query(phrase, biword_index, filenames):
    # tokenize the phrase without stemming
    tokens = word_tokenize(phrase.lower())  # no stemming
    
    # create biwords from the tokenized phrase
    biwords = [" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    
    if not biwords:
        return []

    # start by checking the first biword's results
    result_set = biword_index.get(biwords[0], set())
    
    if not result_set:
        print(f"No documents contain the biword '{biwords[0]}'")
        return []

    # intersect with the results of subsequent biwords
    for biword in biwords[1:]:
        biword_docs = biword_index.get(biword, set())
        result_set = result_set.intersection(biword_docs)

        # if no common documents, return empty
        if not result_set:
            return []

    result_filenames = [filenames[doc_id] for doc_id in result_set]
    return result_filenames


# function to handle proximity queries using positional index
def proximity_query(term1, term2, proximity, positional_index, filenames):
    term1 = stemmer.stem(term1)
    term2 = stemmer.stem(term2)
    
    result_set = set()
    
    docs_with_term1 = positional_index.get(term1, {})
    docs_with_term2 = positional_index.get(term2, {})
    
    for doc_id in docs_with_term1:
        if doc_id in docs_with_term2:
            positions_term1 = docs_with_term1[doc_id]
            positions_term2 = docs_with_term2[doc_id]
            
            for pos1 in positions_term1:
                for pos2 in positions_term2:
                    if abs(pos1 - pos2) <= proximity:
                        result_set.add(doc_id)
                        break
    
    result_filenames = [filenames[doc_id] for doc_id in result_set]
    return result_filenames

# function to handle soundex spelling correction
def soundex_query(query, inverted_index, filenames):
    query = query.lower()
    terms = re.findall(r'\w+', query)
    
    result_set = set()
    operator = None

    for term in terms:
        if term == "and":
            operator = "AND"
        elif term == "or":
            operator = "OR"
        elif term == "not":
            operator = "NOT"
        else:
            soundex_code = soundex(term)
            
            matching_docs = set()
            for inv_term in inverted_index:
                # apply stricter matching----- term should be of similar length
                if soundex(inv_term) == soundex_code and abs(len(inv_term) - len(term)) <= 2:
                    matching_docs = matching_docs.union(inverted_index.get(inv_term, set()))
            
            if not result_set:
                result_set = matching_docs
            else:
                if operator == "AND":
                    result_set = result_set.intersection(matching_docs)
                elif operator == "OR":
                    result_set = result_set.union(matching_docs)
                elif operator == "NOT":
                    result_set = result_set.difference(matching_docs)

            operator = None

    result_filenames = [filenames[doc_id] for doc_id in result_set]
    return result_filenames



# path to the corpus folder containing the .txt files
corpus_folder = "corpus_folder"

# load documents and filenames from the folder
documents, filenames = load_documents_from_folder(corpus_folder)

# build the indexes
inverted_index, biword_index, positional_index = build_indexes(documents)

#----------------------------------------------------------------------------
#---------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# test boolean queries as per test case doc
query1 = "technology OR phone"
query2 = "deliveries AND foods"
phrase_query1 = "search engine"
proximity_query1 = ("easy", "passenger", 10)  # word proximity
soundex_query1 = "yahu and dauwnloads"
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------


# boolean search results
print(f"Results for '{query1}':", boolean_search(query1, inverted_index, filenames))
print(f"Results for '{query2}':", boolean_search(query2, inverted_index, filenames))

# bi-word search results
print(f"Results for phrase query '{phrase_query1}':", phrase_query(phrase_query1, biword_index, filenames))

# proximity search results
print(f"Results for proximity query '{proximity_query1}':", proximity_query(*proximity_query1, positional_index, filenames))

# soundex query results
print(f"Results for soundex query '{soundex_query1}':", soundex_query(soundex_query1, inverted_index, filenames))
