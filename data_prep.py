import os
import json
import zipfile
from datasets import load_dataset
import pandas as pd


def json_to_csv(json_file_path):
    question = []
    context = []
    answer = []

    with open(json_file_path, 'r') as json_file:
    # Read lines from the file
        lines = json_file.readlines()
    
        # Process each line (assuming each line contains a valid JSON dictionary)
        for line in lines:
            # Load the JSON dictionary from the line
            data_dict = json.loads(line)
    
            # Now you can work with the dictionary
            question.append(data_dict['question_refine'])
            answer.append(data_dict['sql'])
            context.append('')

    res = {'question': question, 'answer':answer, 'context':context}

    return pd.DataFrame(res)


if __name__ == '__main__':
    mimicsql_data = 'data_base/mimicsql_data'
    mimicsql_natural = os.path.join(mimicsql_data, 'mimicsql_natural')
    mimicsql_natural_v2 = os.path.join(mimicsql_data, 'mimicsql_natural_v2')
    mimicsql_template = os.path.join(mimicsql_data, 'mimicsql_template')

    train_data = os.path.join(mimicsql_natural, 'train.json')
    test_data = os.path.join(mimicsql_natural, 'test.json')

    train_data = json_to_csv(train_data)
    test_data = json_to_csv(test_data)

    test_data.to_csv(os.path.join(mimicsql_natural, 'test.csv'))
    test_data.to_csv(os.path.join(mimicsql_natural, 'test.csv'))



