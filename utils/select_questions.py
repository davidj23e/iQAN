import argparse
import pickle
import json
import yaml
import os


# Command Line Options
parser=argparse.ArgumentParser(description="Select Questions/Prepare model for FineTuning")

# {train,val}
parser.add_argument("--filter",action='store',type=str,help='select the dataset to be filtered')
parser.add_argument("--replace",action='store',type=str,help='select the dataset to be replaced')
parser.add_argument("--reset",action='store_true',help='reset finetuning')
parser.add_argument("--original",action='store_true',help='use unfiltered VQA dataset')
parser.add_argument("--copy",action='store_true',help='create copy of pre-trained model')


args=parser.parse_args()

# Replaced the model dataset with modified data
# type={train,original}
def replace_data(config,dataset,typet='modified'):    
    if typet=='original':
        os.system("yes | cp -fv "+config['utils_path']+config[dataset+"_"+typet]
        + " "+config['destination']+'/'+dataset+'set.pickle')
    else:
        os.system("mv -v "+config['utils_path']+config[dataset+"_"+typet]
        + " "+config['destination']+'/'+dataset+'set.pickle')
    print('Replaced',dataset,'data !')

# Restore the pre-trained model
def reset(config):
    os.system("rm -rf "+config['model_folder'])
    os.system("cp -Rv "+config['utils_path']+config['pre_trained']+"/* "+config['model_parent'])
    print('Restored pre-trained config')

def copy_data(config):
    os.system("rm -rf "+config['utils_path']+config['pre_trained']+"/*")
    os.system("cp -Rv "+config['model_folder']+" "+config['utils_path']+config['pre_trained'])
    print('Copied trained model to pre-train directory')

# Filtering Question from VQA dataset
# dataset = {train,val}
def filter_questions(config, dataset):
    path = config['utils_path']
    if(dataset == 'train'):
        path += config['train_qid']

        f = open(path, 'r')
        qids = json.load(f)

        fp = open(config['utils_path']+config['train_original'], 'rb')
        data = pickle.load(fp)

        # Filter Question IDS
        mod_data = []
        count = 0
        for element in data:
            if(element['question_id'] in qids):
                count += 1
                mod_data.append(element)
                print("\rTrainset Count : ", count, end="")
        print()

        f.close()
        fp.close()

        f = open(config['utils_path']+config['train_modified'], "wb")
        pickle.dump(mod_data, f)
        f.close()

    if(dataset == 'val'):
        path += config['val_qid']

        f = open(path, 'r')
        qids = json.load(f)

        fp = open(config['utils_path']+config['val_original'], 'rb')
        data = pickle.load(fp)

        # Filter Question IDS
        mod_data = []
        count = 0
        for element in data:
            if(element['question_id'] in qids):
                count += 1
                mod_data.append(element)
                print("\rValset Count : ", count, end="")
        print()

        f.close()
        fp.close()

        f = open(config['utils_path']+config['val_modified'], "wb")
        pickle.dump(mod_data, f)
        f.close()


if __name__ == "__main__":
    path = open('utils/config.yaml', 'r')
    config = yaml.safe_load(path)

    # create directories
    os.system("mkdir -p "+config['utils_path']+"/""pickle_modified")
    os.system("mkdir -p "+config['utils_path']+"/"+"pickle_dataset")
    os.system("mkdir -p "+config['utils_path']+"/"+"qid")
    os.system("mkdir -p "+config['utils_path']+"/"+"pre_trained_model")

    # Filter dataset with qids
    if(args.filter is not None):
        filter_questions(config,args.filter)

    # Replace dataset with modified dataset
    if(args.replace is not None):
        replace_data(config,args.replace)

    # Use unfiltered VQA 
    if(args.original):
        replace_data(config,'train','original')
        replace_data(config,'val','original')
    
    if(args.copy):
        copy_data(config)

    print(args)
    # Restore pretrained model
    if(args.reset):
        reset(config)
