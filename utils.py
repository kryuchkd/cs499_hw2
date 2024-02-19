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

















from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
def plot_scatter_and_fit(occurance_list):
    
    # Create x-axis values
    x = range(1, len(occurance_list) + 1)

    # Create y-axis values
    y = occurance_list

    def power_law(x, a, b):
        return a * x + b

    params, covariance = curve_fit(power_law, np.log(x), np.log(y))

    slope = params[0]
    intercept = params[1]
    print(f"Slope of the linear relationship: {slope}")
    print(f"Intercept of the linear relationship: {intercept}")

    plt.scatter(np.log(x), np.log(y))
    plt.xlabel('log2(Rank)')
    plt.ylabel('log2(Frequency)')
    plt.title('Log-Log Plot of Word Frequencies')
    x_fit = np.linspace(min(np.log(x)), max(np.log(y)))
    y_fit = power_law(x_fit, *params)
    plt.plot(x_fit, y_fit, color='red', label=f'Fitted Line (Slope: {slope:.2f})')
    plt.legend()

    # Show the plot
    plt.show()

def plot_scatter_and_fit_2(occurance_list):
    x = range(1, len(occurance_list) + 1)
    y = occurance_list

    #calculate the slope and x and y intercepts of the line of best fit
    slope, intercept = np.polyfit(x, y, 1)

    #plot the scatter plot and the line of best fit on the log-log scale
    plt.scatter(x, y)
    plt.plot(x, slope*x + intercept, color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Log-Log Plot of Word Frequencies')
    plt.show()
    print(f"Slope: {slope}")
    print(f"Intercept: {intercept}")
    

def count_ducument_tokens(list_of_docs):
    # Create a defaultdict to store token occurrences
    token_counts = defaultdict(int)

    # Iterate through the nested lists
    for current_file_tokenized_reviews in list_of_docs:
        for review in current_file_tokenized_reviews:
            for token in set(review):
                token_counts[token] += 1

    # Convert defaultdict to a regular dictionary
    token_counts_dict = dict(token_counts)

    return token_counts_dict

import numpy as np
import matplotlib.pyplot as plt

def loglog_scatter_with_fit(x_values, y_values):
    """
    Create a log-log scatter plot with a line of best fit.

    Parameters:
    - x_values (list or array): List of x values.
    - y_values (list or array): List of y values.

    Returns:
    - None
    """
    if len(x_values) != len(y_values):
        raise ValueError("Lengths of x_values and y_values must be equal.")

    # Convert input lists to numpy arrays
    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # Create log-log scatter plot
    plt.loglog(x_values, y_values, 'o', label='Data Points')

    # Perform linear regression to find the line of best fit
    slope, intercept = np.polyfit(np.log(x_values), np.log(y_values), 1)

    # Generate the line of best fit
    fit_line = np.exp(intercept) * x_values**slope

    # Plot the line of best fit
    plt.loglog(x_values, fit_line, label=f'Line of Best Fit: y = {np.exp(intercept):.2f} * x^{slope:.2f}')

    # Display the slope and intercept
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {np.exp(intercept):.4f}")

    # Add labels and legend
    plt.xlabel('Log(x)')
    plt.ylabel('Log(y)')
    plt.legend()

    # Show the plot
    plt.show()