import argparse
import cv2
import h5py
import numpy as np
import os
from tqdm import tqdm
import datetime

from sklearn.cluster import KMeans

from dms import geometry
from dms.dataloader import EvaluationDataset
from dms.io import recursive_write_h5


def parse_args():
    parser = argparse.ArgumentParser('Run keypoint clustering.')
    parser.add_argument('--dataset', type=str, default='megadepth', choices=['megadepth', 'scannet', 'wxbs' ])
    parser.add_argument('--keypoints', type=str, default='dkm', choices=['dkm', 'roma', 'aspanformer', 'mast3r'])
    parser.add_argument('--method', type=str, default='kmeans4d', choices=['slic', 'kmeans2d', 'kmeans4d', 'kmeans9d', 'grid'])
    parser.add_argument('--num_components', type=int, default=128)
    parser.add_argument('--no_save_assignment', action='store_true')
    return parser.parse_args()


def assign_matches_to_clusters(clusters, assignment, x1, x2, prune_empty=False):
    num_points = len(x1)

    for i in range(num_points):

        # Get id of assigned cluster from first image
        coords = x1[i].astype(int)
        cluster_id = assignment[coords[1], coords[0]]
        
        clusters[cluster_id].setdefault('x1', []).append(x1[i])
        clusters[cluster_id].setdefault('x2', []).append(x2[i])

    for c in clusters:
        if 'x1' not in c:
            continue
        c['x1'] = np.array(c['x1'])
        c['x2'] = np.array(c['x2'])  

    if prune_empty:
        # Only return nonempty clusters
        return [c for c in clusters if 'x1' in c and len(c['x1']) > 0]
    else:
        return clusters


def cluster_list_to_dict(clusters):
    return {str(c['number']): c for c in clusters}


def run_slic(slic, image_pair):

    # Load image
    (im1_path, im2_path) = image_pair.image_paths()
    im1 = cv2.imread(im1_path)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

    # Run SLIC
    assignment = slic.iterate(im1)
    slic_clusters = slic.slic_model.clusters

    # Assign matches to clusters
    (x1, x2) = image_pair.matches()
    clusters = assign_matches_to_clusters(slic_clusters, assignment, x1, x2)
    clusters = cluster_list_to_dict(clusters)

    return clusters, assignment


def run_kmeans(kmeans_estimator, image_pair, method='2D', calibrated=True):
    
    # Clustering can optionally be done on calibrated matches (defaults to True)
    (x1, x2) = image_pair.matches(calibrated=calibrated)

    if method == '2D':
        features = x1
    elif method == '4D':
        features = np.c_[x1, x2]
    elif method == '9D':
        features = geometry.setup_fundamental_dlt_matrix(x1, x2)
    else:
        raise ValueError(f'Unknown KMeans method: {method}.')
    
    if len(features) < kmeans_estimator.get_params()['n_clusters']:
        return None

    kmeans_clusters = kmeans_estimator.fit(features)

    # Always return uncalibrated matches
    (x1_pixels, x2_pixels) = image_pair.matches()

    n_clusters = kmeans_clusters.get_params()['n_clusters']
    clusters = []
    for k in range(n_clusters):
        labels = (kmeans_clusters.labels_ == k)
        clusters.append({
            'x1': x1_pixels[labels],
            'x2': x2_pixels[labels],
            'num_points': np.sum(labels),
            'number': k,
        })

    clusters = cluster_list_to_dict(clusters)
    return clusters



def run_grid_pts(x1, x2, k):
    n = np.ceil(np.sqrt(k)).astype(int)
 
    min_coord = np.min(x1, axis=0)
    max_coord = np.max(x1, axis=0)
    shape = max_coord - min_coord + 1
 
    # put points in grid
    grid = [[[] for _ in range(n)] for _ in range(n)]
    
    tt1 = datetime.datetime.now()
    for i in range(len(x1)):
        x, y = x1[i]
 
        ind_i = int((x - min_coord[0]) / shape[0] * n)
        ind_j = int((y - min_coord[1]) / shape[1] * n)
        grid[ind_i][ind_j].append((x1[i], x2[i]))
    tt2 = datetime.datetime.now()
    rt = (tt2 - tt1).total_seconds()
 
    clusters = []
    cluster_number = 1
    for i in range(n):
        for j in range(n):
            if len(grid[i][j]) > 0:
                x1, x2 = zip(*grid[i][j])
                x1 = np.array(x1)
                x2 = np.array(x2)
                clusters.append({
                    'number': cluster_number,
                    'x1': x1,
                    'x2': x2,
                    'num_points': len(x1)
                })
                cluster_number += 1
 
    return clusters, rt


def run_grid_clustering(k, image_pair):
    (x1, x2) = image_pair.matches()
    clusters = run_grid_pts(x1, x2, k)[0]
    clusters = cluster_list_to_dict(clusters)
    return clusters


def save_clusters(clusters, assignment, image_pair, save_path, save_assignment=True):
    # Add clusters to image_pair dataset
    # Optionally save assignment
    with h5py.File(save_path, 'a') as f:
        f.copy(image_pair.data, f)
        recursive_write_h5(f[image_pair.key], {'clusters': clusters})

        if save_assignment:
            recursive_write_h5(f[image_pair.key], {'assignment': assignment})


def main(args):
    save_assignment = not args.no_save_assignment

    if args.dataset in ['megadepth', 'scannet']:
        dataset_name = args.dataset + '1500'
        image_path = f'data/{args.dataset}1500-images/images'
        calibrated = True
    elif args.dataset == 'wxbs':
        dataset_name = 'wxbs'
        image_path = f'.WxBS/v1.1'
        calibrated = False

    data_path = f'data/{dataset_name}_{args.keypoints}.h5'
    dataset = EvaluationDataset(data_path, image_path)

    save_path = f'data/clustered/{dataset_name}_{args.keypoints}_{args.method}{args.num_components}.h5'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        print(f"Clusters already exist for {args.dataset}, {args.keypoints}, {args.method}, {args.num_components}")
        return
    print(f"Results will be saved to {save_path}")

    if args.method == 'slic':
        raise NotImplementedError("SLIC support is omitted from code release due to dependency issues. If want to use it, check out https://github.com/Algy/fast-slic.")
        # slic = Slic(num_components=args.num_components, compactness=10)
        # for image_pair in tqdm(dataset, desc='Running SLIC'):
        #     clusters, assignment = run_slic(slic, image_pair)
        #     save_clusters(clusters, assignment, image_pair, save_path, save_assignment)

    elif args.method.startswith('kmeans'):
        kmeans_estimator = KMeans(n_clusters=args.num_components, random_state=0)
        dim = f'{args.method[-2]}D'
        for image_pair in tqdm(dataset, desc='Running KMeans'):
            clusters = run_kmeans(kmeans_estimator, image_pair, dim, calibrated)
            save_clusters(clusters, None, image_pair, save_path, save_assignment)

    elif args.method == 'grid':
        for image_pair in tqdm(dataset, desc='Running Grid'):
            clusters = run_grid_clustering(args.num_components, image_pair)
            save_clusters(clusters, None, image_pair, save_path, save_assignment)

    else:
        raise ValueError(f'Unknown clustering method {args.method}')


if __name__ == "__main__":
    args = parse_args()
    main(args)
