import re
from sklearn.feature_extraction.text import CountVectorizer

def find_common_words(all_words, num_most_frequent_words):
    vectorizer = CountVectorizer(
        stop_words=None, # 'english',
        max_features=num_most_frequent_words,
        binary=True)
    vectorizer.fit(all_words)
    return (vectorizer.vocabulary_, vectorizer.get_feature_names())


def read_odyssey_tuples(tuplesize,
                        num_most_frequent_words,
                        verbose=False):
    with open('pg1727-part.txt', 'r') as file:
        text = re.findall(r'[a-zA-Z]+', file.read())

    (common_voc, common_names) = find_common_words(text, num_most_frequent_words)
    print(common_voc)
    print(common_names)

    res = []
    dist = 12
    for i in range(len(text)-dist):
        first_word = text[i]
        if first_word in common_voc:
            a = common_voc[first_word]
            tuple = [a]
            for j in range(dist):
                next_word = text[i+1+j]
                if next_word in common_voc:
                    n = common_voc[next_word]
                    tuple.append(n)
                    if len(tuple) == tuplesize:
                        res.append(tuple)
                        if verbose and i < 200:
                            print(tuple)
                            print('from ', text[i:i+2+j])
                        break
    return (res, common_names)
    

if __name__ == "__main__":
    num_words = 20
    (tuples, words) = read_odyssey_tuples(3, num_words, verbose=True)
    print('number of common word tuples: ', len(tuples))
    for s in range(10):
        for i in tuples[s]:
            print(i, words[i])
        print('')
    ts = set([(a,b,c) for a,b,c in tuples])
    print('distinct word tuples: ', len(ts))
