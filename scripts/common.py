import json
import gzip
import array
import pickle
import numpy as np
import random
import pandas as pd
import time
from tqdm import tqdm

def readImageFeatures(path):
    f = open(path, 'rb')
    while True:
        asin = f.read(10)
        if asin == '':
            break
        a = array.array('f')
        a.fromfile(f, 4096)
        yield asin, a.tolist()

def parse(path):
    g = open(path, 'r')
    for l in g:
        raw = l.split()
        data = {'uid':raw[0], 'pid':raw[1], 'rating':float(raw[2])}
        yield data

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def reverse_(D):
    nD = {}
    for key in D:
        vals = D[key]
        for val, rate in vals:
            try:
                nD[val].append((key, rate))
            except:
                nD[val] = []
                nD[val].append((key, rate))
    for key in nD:
        nD[key] = list(set(nD[key]))
    return nD

def remove_cold_start(prod_dict, user_dict):
    random.seed(7)
    len_dict1 = {prod:prod_dict[prod] for prod in prod_dict if (len(prod_dict[prod]) < 25 and len(prod_dict[prod]) >= 10)}
    random.seed(7)
    len_dict2 = {prod:prod_dict[prod] for prod in prod_dict if (len(prod_dict[prod]) < 50 and len(prod_dict[prod]) >= 25)}
    keys1 = random.sample(list(len_dict1.keys()), 30)
    keys2 = random.sample(list(len_dict2.keys()), 20)
    cold_start, new_prod_dict = {}, {}
    for prod in prod_dict:
        if ((prod in keys1) or (prod in keys2)):
            cold_start[prod] = prod_dict[prod]
        else:
            new_prod_dict[prod] = prod_dict[prod]
    new_user_dict = reverse_(new_prod_dict)
    return cold_start, new_prod_dict, new_user_dict

def Helper(cold_start, new_user_dict):
    n, tot = 0, 0
    for p in cold_start:
        for user, rating in cold_start[p]:
            tot += 1
            try:
                new_user_dict[user]
            except:
                n += 1
    return n, tot

def prepare_data(path_to_review_file, dataset_name):
    try:
        cold_start = pickle.load(open('../saved/{}/{}_cold_start.pkl'.format(dataset_name, dataset_name), 'rb'))
        user_dict = pickle.load(open('../saved/{}/{}_user_dict.pkl'.format(dataset_name, dataset_name), 'rb'))
        prod_dict = pickle.load(open('../saved/{}/{}_prod_dict.pkl'.format(dataset_name, dataset_name), 'rb'))
        return cold_start, prod_dict, user_dict
    except:
        start = time.time()
        logfile = open('../log/'+dataset_name+'_log.txt', 'w')
        logfile.write('\n-----READING DATA-----\n')
        review_data = getDF(path_to_review_file)
        user_dict_p = {u: list(p) for u, p in review_data.groupby('uid')['pid']}
        user_dict_r = {u: list(r) for u, r in review_data.groupby('uid')['rating']}
        logfile.write('Raw #users = {}\n'.format(len(user_dict_p)))
        # print('Raw #users = {}\n'.format(len(user_dict_p)))
        user_dict = {}
        for user in user_dict_p:
            if len(user_dict_p[user]) < 2:
                continue
            user_dict[user] = list(zip(user_dict_p[user], user_dict_r[user]))
        logfile.write('#users with > 1 interactions = {}\n'.format(len(user_dict)))
        # print('#users with > 1 interactions = {}\n'.format(len(user_dict)))
        user_dict[user] = list(zip(user_dict_p[user], user_dict_r[user]))
        prod_dict = reverse_(user_dict)
        logfile.write('#prods = {}\n'.format(len(prod_dict)))
        # print('#prods = {}'.format(len(prod_dict)))
        cold_start, new_prod_dict, new_user_dict = remove_cold_start(prod_dict, user_dict)
        logfile.write('#cold start products = {}\n'.format(len(cold_start)))
        n, tot = Helper(cold_start, new_user_dict)
        logfile.write('#users in cold products = {}\n'.format((tot)))
        logfile.write('#users in cold products who are not present in user_dict = {}\n'.format((n)))
        logfile.write('#users after removing Cold Start Products = {}\n'.format(len(new_user_dict)))
        
        logfile.write('#products after removing Cold Start Products = {}\n'.format(len(new_prod_dict)))
        
        end = time.time()
        elapsed = end - start
        logfile.write('Time taken for reading = {:.4f} sec\n'.format(elapsed))
        logfile.close()
    return cold_start, new_prod_dict, new_user_dict

def refine(user_dict, prod_dict, cold_start, prod_images, cold_images, dataset_name):
    ref_cold, ref_user, ref_prod = {}, {}, {}
    ref_cold = {pid:cold_start[pid] for pid in cold_images}
    ref_prod = {pid:prod_dict[pid] for pid in prod_images}
    ref_user = reverse_(ref_prod)
    return ref_cold, ref_user, ref_prod

def prepare_image_data(user_dict, prod_dict, cold_start, image_path, dataset_name):
    try:
        cold_images = pickle.load(open('../saved/{}/{}_cold_images.pkl'.format(dataset_name, dataset_name), 'rb'))
        prod_images = pickle.load(open('../saved/{}/{}_prod_images.pkl'.format(dataset_name, dataset_name), 'rb'))
    except:
        start = time.time()
        cold_images = {}
        prod_images = {}
        try:
            for pid, image in tqdm(readImageFeatures(image_path)):
                pid = pid.decode('ascii')
                try:
                    tmp = prod_dict[pid]
                    prod_images[pid] = image
                except:
                    pass
                try:
                    tmp = cold_start[pid]
                    cold_images[pid] = image
                except:
                    pass
        except EOFError:
            print('File Read')
        end = time.time()
        elapsed = end - start
        with open('../log/'+dataset_name+'_log.txt', 'a') as logfile:
            logfile.write('\nTime taken for reading images = {:.4f} sec\n'.format(elapsed))
        with open('../saved/{}/{}_cold_images.pkl'.format(dataset_name, dataset_name), 'wb') as file:
            pickle.dump(cold_images, file)
        with open('../saved/{}/{}_prod_images.pkl'.format(dataset_name, dataset_name), 'wb') as file:
            pickle.dump(prod_images, file)

        cold_start, user_dict, prod_dict = refine(user_dict, prod_dict, cold_start, prod_images, cold_images, dataset_name)
        with open('../saved/{}/{}_cold_start.pkl'.format(dataset_name, dataset_name), 'wb') as file:
            pickle.dump(cold_start, file)
        with open('../saved/{}/{}_user_dict.pkl'.format(dataset_name, dataset_name), 'wb') as file:
            pickle.dump(user_dict, file)
        with open('../saved/{}/{}_prod_dict.pkl'.format(dataset_name, dataset_name), 'wb') as file:
            pickle.dump(prod_dict, file)
        with open('../log/'+dataset_name+'_log.txt', 'a') as logfile:
            logfile.write('\nFinal stats after removing missing image files:\n')
            logfile.write('#users = {}\n #products = {}\n #cold_start = {}\n'.format(len(user_dict), len(prod_dict), len(cold_start)))
    return prod_images, cold_images

if __name__ == '__main__':
    cold_start, prod_dict, user_dict = prepare_data('../raw_data/Baby/reviews_Baby.votes', 'Baby')
    prod_images, cold_images = prepare_image_data(user_dict, prod_dict, cold_start, '../raw_data/Baby/image_features_Baby.b', 'Baby')
