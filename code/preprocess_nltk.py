import time
import sys
import nltk
import multiprocessing
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def load_data(path):
    print('Loding data...')
    data = pd.read_csv(path, encoding='utf-8')
    docs = list(data['content'])
    return docs

#show progress bar
def progress_bar(total,current):
    max_step = 50
    num_arrow = int(current/total*50)
    num_dash = max_step - num_arrow
    bar = '[' + '#'*num_arrow + '-'*num_dash + ']' + '{:.2f}%'.format(current/total*100) + '\r'
    sys.stdout.write(bar)
    sys.stdout.flush()

#multiple process
def muti_call(func,docs):
    count = 0
    for i in pool.imap(func,iter(docs)):
        docs[count] = i
        count+=1
        progress_bar(len(docs),count)
    print()

#test if it is a float or int type
def isnumber(str):
    try:
        float(str)
        return True
    except:
        return False

#Tokenize words
#nltk.help.upenn_tagset() to show the meaning of each pos_tag
def token_words(sents):
    for i in range(len(sents)):
        sents[i] = nltk.pos_tag(word_tokenize(sents[i]))
    return sents

if __name__ == '__main__':
    #Build mutiprocess environment
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cores)

    st = time.time()

    #load original data
    path = '../data/r5/output/data.csv'
    docs = load_data(path)
    docs = docs[0:5]

    #Tokenize sentence
    print('tokenize sentence...')
    muti_call(sent_tokenize,docs)
    
    #Tokenize words and add pos tags
    print('tokenize words and add pos_tags...')
    muti_call(token_words,docs)

    for i in docs[0]:
        print(i)
    
    runtime = time.time() - st
    print('Total running time: {} seconds'.format(runtime))