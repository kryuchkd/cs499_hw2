import json
import nltk
import os
import re
from string import punctuation

def get_all_tokenized_reviews():
    list_of_reviews = []
    ps = nltk.stem.PorterStemmer()
    #go through all files
    for doc_num, file_path in enumerate(os.listdir('yelp')):
        #open each file
        with open('yelp/'+file_path, 'r') as file:
            #load json into a python dict
            file_as_dict = json.load(file)
            #get a list of individual reviews (just their content)
            for review in file_as_dict['Reviews']:
                #tokenize the review using nltk
                current_review_tokenized = nltk.tokenize.word_tokenize(review['Content'])
                #to lowercase and remove punctuation
                new_arr = []
                for word in current_review_tokenized:
                    #Avoid cases like 'Squid.We', etc. split each token on punctuation except for '
                    escaped_punctuation = re.escape(punctuation.replace("'", ""))

                    # Define a regular expression pattern to match the escaped custom punctuation characters
                    punctuation_pattern = f'[{escaped_punctuation}]'

                    # Use re.split to split the string based on the custom punctuation pattern
                    parts = re.split(punctuation_pattern, word)

                    # Filter out empty strings from the result
                    result = [part for part in parts if part]

                    #split tokens and non split tokens get handled the same
                    for word in result:
                        #remove leftorve punctuation
                        word = word.translate(str.maketrans('', '', punctuation))
                        #empty sting means it was just a punctuation token
                        if word != '':
                            if word.isdigit():
                                #replace numbers with NUM as per instructions
                                new_arr.append('NUM')
                            else:
                                #if its a word, stem it
                                new_arr.append(ps.stem(word))
                            #save the tokenized review
                list_of_reviews.append(new_arr)
    return list_of_reviews

from collections import defaultdict

def total_term_frequency(list_of_tokenized_reviews):
    # Create a defaultdict to store number of total token occurrences
    token_counts = defaultdict(int)

    # Iterate through the nested lists
    for review in list_of_tokenized_reviews:
        for token in review:
            # Increment the count for each token
            token_counts[token] += 1

    # Convert defaultdict to a regular dictionary
    token_counts_dict = dict(token_counts)

    return token_counts_dict

def get_all_bigrams(list_of_tokenized_reviews):
    all_bigrams = []
    for review in list_of_tokenized_reviews:
        for i in range(len(review)-1):
            all_bigrams.append((review[i], review[i+1]))
    return all_bigrams


def total_bigram_freqency(all_bigrams):
    bigram_counts = defaultdict(int)

    # Iterate through the nested lists
    for bigram in all_bigrams:
        # Increment the count for each bigram
        bigram_counts[bigram] += 1

    return dict(bigram_counts)

def get_all_tokenized_reviews_2():
    list_of_reviews = []
    #go through all files
    for doc_num, file_path in enumerate(os.listdir('yelp')):
        #open each file
        with open('yelp/'+file_path, 'r') as file:
            #load json into a python dict
            file_as_dict = json.load(file)
            #get a list of individual reviews (just their content)
            for review in file_as_dict['Reviews']:
                #tokenize the review using nltk
                current_review_tokenized = nltk.tokenize.word_tokenize(review['Content'])
                #to lowercase and remove punctuation
                new_arr = []
                for word in current_review_tokenized:
                    #Avoid cases like 'Squid.We', etc. split each token on punctuation except for '
                    escaped_punctuation = re.escape(punctuation.replace("'", ""))

                    # Define a regular expression pattern to match the escaped custom punctuation characters
                    punctuation_pattern = f'[{escaped_punctuation}]'

                    # Use re.split to split the string based on the custom punctuation pattern
                    parts = re.split(punctuation_pattern, word)

                    # Filter out empty strings from the result
                    result = [part for part in parts if part]

                    #split tokens and non split tokens get handled the same
                    for word in result:
                        #remove leftorve punctuation
                        word = word.translate(str.maketrans('', '', punctuation_pattern))
                        #empty sting means it was just a punctuation token
                        if word != '':
                            if word.isdigit():
                                #replace numbers with NUM as per instructions
                                new_arr.append('NUM')
                            else:
                                #if its a word, stem it
                                new_arr.append(word.lower())
                            #save the tokenized review
                list_of_reviews.append(new_arr)
    return list_of_reviews