import re
import pandas as pd



def txt_to_lists (dir):
    handle = open(dir, "r")
    datal = handle.readlines() 
    data = ""
    for one in datal:
        data += one
    data = re.split("\t|\n", data)
    data = data[:-1]
    labels, examples = [], []
    i = 0
    for one in data:
        if i%2 == 0:
            if one == "No":
                labels.append(0)
            else:
                labels.append(1)
        else:
            examples.append(one)
        i += 1
    handle.close()
    return (labels, examples)


train_labels, train_examples = txt_to_lists("data/trainSet")
test_labels, test_examples = txt_to_lists("data/testSet")
#train_data = pd.DataFrame(data = {'id': range(len(train_labels)), 'label': train_labels, 'alpha': ['a']*len(train_labels), 'example': train_examples})
#test_data = pd.DataFrame(data = {'id': range(len(test_labels)), 'label': test_labels, 'alpha': ['a']*len(test_labels), 'example': test_examples})
#train_data.to_csv('data/train.tsv', sep='\t', index=False, header=False)
#test_data.to_csv('data/dev.tsv', sep='\t', index=False, header=False)