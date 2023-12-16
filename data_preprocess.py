import json, os, glob
import _pickle as cPickle
from collections import defaultdict




def load_stats_train(data_dir, dataset,list_name):
    img_path = os.path.join( dataset,list_name )
    
    img_list = [os.path.join(data_dir,img_path.split('/')[0], line.rstrip('\n'))
                        for line in open(os.path.join(data_dir, img_path))]
    dict_cat = defaultdict(list)
    for img in img_list:
        with open(img + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
            for instance,cls  in enumerate(gts['class_ids']):
                dict_cat[cls].append((img, instance))
    
    return dict_cat

def load_stats_test(data_dir, dataset):
    result_pkl_list = glob.glob(os.path.join(data_dir, 'detection', dataset, 'results_*.pkl'))
    dict_cat = defaultdict(list) 
    
    for path in result_pkl_list:
        with open(path, 'rb') as f:
            pred_data = cPickle.load(f)
        # image_path = os.path.join(data_dir, pred_data['image_path'][5:])
        for instance,cls  in enumerate(pred_data['pred_class_ids']):
            dict_cat[int(cls)].append((path, instance))
    return dict_cat

data_dir = "/media/student/Data/yamei/data/NOCS/"

camera_train_stats = load_stats_train(data_dir, 'camera', 'train_list.txt')
with open(os.path.join(data_dir, 'camera', 'train_category_dict.json'), 'w') as fp:
    json.dump(camera_train_stats, fp)
real_train_stats = load_stats_train(data_dir, 'real', 'train_list.txt')

with open(os.path.join(data_dir, 'real', 'train_category_dict.json'), 'w') as fp:
    json.dump(real_train_stats, fp)


camera_test_stats = load_stats_test(data_dir, 'CAMERA25')
import pdb;pdb.set_trace()
with open(os.path.join(data_dir, 'detection', 'camera_test_category_dict.json'), 'w') as fp:
    json.dump(camera_test_stats, fp)

real_test_stats = load_stats_test(data_dir, 'REAL275')
with open(os.path.join(data_dir, 'detection', 'real_test_category_dict.json'), 'w') as fp:
    json.dump(real_test_stats, fp)
