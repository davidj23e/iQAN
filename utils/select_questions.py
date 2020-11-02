import argparse
import pickle
import json
import yaml
import os


# Replaced the model dataset with modified data
# type={train,original}
def replace_data(config, type='modified'):
    os.system("mv "+config['utils_path']+config['train_'+type]
              + " "+config['destination']+'trainset.pickle')
    os.system("mv "+config['utils_path']+config['val_'+type]
              + " "+config['destination']+'valset.pickle')


# Filtering Question from VQA dataset
# dataset = {train,val}
def filter_questions(config, dataset):
    path = config['utils_path']
    if(dataset == 'train'):
        path += config['train_qid']

        f = open(path, 'r')
        qids = json.load(f)

        fp = open(config['train_original'], 'rb')
        data = pickle.load(fp)

        mod_data = []
        count = 0
        for element in data:
            if(element['question_id'] in qids):
                count += 1
                mod_data.append(element)
                print("\r Count : ", count, end="")
        print("\n")

        f.close()
        fp.close()

        f = open(config['utils_path']+config['train_modified'], "wb")
        pickle.dump(mod_data, f)
        f.close()

    if(dataset == 'val'):
        path += config['val_qid']

        f = open(path, 'r')
        qids = json.load(f)

        fp = open(config['val_original'], 'rb')
        data = pickle.load(fp)

        mod_data = []
        count = 0
        for element in data:
            if(element['question_id'] in qids):
                count += 1
                mod_data.append(element)
                print("\r Count : ", count, end="")
        print("\n")

        f.close()
        fp.close()

        f = open(config['utils_path']+config['val_modified'], "wb")
        pickle.dump(mod_data, f)
        f.close()


if __name__ == "__main__":
    path = open('config.yaml', 'r')
    config = yaml.safe_load(path)

    # create directories
    os.system("mkdir -p "+config['utils_path']+"\\"+"pickle_modified")
    os.system("mkdir -p "+config['utils_path']+"\\"+"pickle_dataset")
    os.system("mkdir -p "+config['utils_path']+"\\"+"qid")

    filter_questions(config, 'train')
    filter_questions(config, 'val')
    # replace_data(config)
