from glob import glob
import numpy as np
import warnings
import pandas as pd
import time
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from scipy import linalg as la
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from bpCamera import *
from bpSingleImage import *
from PhotoViewer import *
from misc import *
from tqdm import tqdm
from prettytable import PrettyTable

pd.set_option('display.precision', 5)

if __name__ == '__main__':
    startTime = time.time()
    # setting initial camera parameters
    f = 150  # mm
    avg_z = 37.37  # m
    xp = 0  # m
    yp = 0  # m
    sensor_size = 5.6e-3  # mm
    meter_per_pixel = 0.025  # m
    imageWidth_pix, imageHeight_pix = 16768., 14016.
    imageWidth, imageHeight = imageWidth_pix * meter_per_pixel, imageHeight_pix * meter_per_pixel

    eop_path = r"raw_data\wide_format\eop_full_block.txt"
    points_path = r"raw_data\wide_format\cpoints\*.txt"

    # creating camera object
    a = imageWidth_pix * sensor_size / 2
    b = imageHeight_pix * sensor_size / 2
    fiducials = np.array([[-a, -b], [a, b], [-a, b], [a, -b]])
    # define 'fiducials' for images
    fiducials_image = np.array([[-0.5, imageHeight_pix - 0.5], [imageWidth_pix - 0.5, -0.5], [-0.5, -0.5],
                                [imageWidth_pix - 0.5, imageHeight_pix - 0.5]])  # pixels
    cam = Camera(f, [xp, yp], fiducials, sensor_size)

    # read & define images
    data = pd.read_table(eop_path, index_col=0)
    images = []
    for dat in data.iterrows():
        # define img object
        img = SingleImage(dat[-1]['imageName'], cam)
        # define starting values for exterior orientation
        img.exteriorOrientationParameters = np.hstack(
            (np.array(dat[-1][1:4]), np.radians(np.array(dat[-1][4:None]).astype(float)))).astype(float)
        # compute inner orientation (pix -> mm) for img
        img.ComputeInnerOrientation(fiducials_image)
        # append to list of imgs
        images.append(img)

    # extract image names
    image_names = pd.DataFrame(np.arange(data.iloc[:, 0].size), index=data.iloc[:, 0])

    # load tie and control points sample in every image
    #
    # eventually we should get to -
    # res = non_linear_LSA(cam, images, all_appx_ground_points, images_sampled_points, epsilon=1e-4, iters=10, LM=False)
    # where cam&images are ready. all_appx_ground_points is a df with every computed ground point
    # images_sampled_points is a list of df's with every sample in every image in mm
    #########################################################

    file_names = glob(points_path)
    data = []
    cp_file_names = []
    image_ids = []
    for i, f_name in enumerate(file_names):
        with open(f_name, 'r', encoding='UTF-8') as file:
            cp_file_names.append(f_name.split('\\')[-1].split('.')[0])  # [0].split('+'))
            lines = file.readlines()
            cpoints_imgs = []
            for line in lines:
                if line != lines[0]:
                    line = line.split()
                    cpoints_imgs.append(line)
        if f_name != file_names[-1]:
            cpoints_imgs = np.vstack(cpoints_imgs)
            cpoints_imgs[:, [1, 2]] = cpoints_imgs[:, [2, 1]]
            # cpoints_imgs[:, -1] = -1 * cpoints_imgs[:, -1]
            data.append(cpoints_imgs)
            image_ids.append(tuple(cp_file_names[i].split('+')))

    # load gcp
    with open(file_names[-1], 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        gcp = []
        for line in lines:
            line = line.split()
            gcp.append(line)
    gcp = np.vstack(gcp)
    gcp = pd.DataFrame(gcp[:, 1:None], index=gcp[:, 0])

    # del trash from memory
    del dat, img, file, file_names, line, lines

    ####
    # getting z values for fixing
    tie_ground_values = pd.read_csv(r'raw_data\wide_format\tie_vals.txt', header=None, index_col=0).sort_index()
    ####

    # convert image samples into camera system and to ground system
    appx_ground_data = []
    for i, dat in enumerate(data):
        # converting to float and adjusting dataframe
        # data[i] = data[i].drop_duplicates()
        data[i] = pd.DataFrame(dat[:, 1:], index=dat[:, 0]).astype(float).sort_index().drop_duplicates()
        #
        # fixing pixel samples to make y axis positive
        data[i].iloc[:, -1] = -1 * data[i].iloc[:, -1]
        #
        img_id = cp_file_names[i]
        #
        # no need to convert pixel samples like med_format. just multiply by sensor size
        data[i].loc[:, :1] = sensor_size * data[i].loc[:, :1]
        # data[i].loc[:, :1] = images[int(image_names.loc[img_id])].ImageToCamera(data[i].loc[:, 0:1])
        #

        # np.full(data[i].values.shape[0], avg_z)
        # tie_ground_values[tie_ground_values.index.isin(data[i].index)].iloc[:, -1].values

        # get points in ground system by geometric intersection
        appx_ground_data = ImageToGround_GivenZ(
            np.hstack((data[i].values, np.full(data[i].values.shape[0], -f)[:, None])),
            np.full(data[i].values.shape[0], avg_z), images[int(image_names.loc[img_id])].exteriorOrientationParameters,
            f)

        # appx_ground_data = tie_ground_values[tie_ground_values.index.isin(data[i].index)].sort_index().drop_duplicates()

        appx_ground_data = pd.DataFrame(appx_ground_data, index=data[i].index)

        data[i] = pd.concat([data[i], appx_ground_data], axis=1)
        data[i].columns = ['x1[mm]', 'y1[mm]', 'X[m]', 'Y[m]', 'Z[m]']
        data[i] = data[i].sort_index()

    # retrieving all points id's
    all_points_id = []
    for i, dat in enumerate(data):
        all_points_id.append(np.array(dat.index))

    all_sampled_points_id = np.concatenate(all_points_id, axis=0)
    # all_sampled_points_id = np.unique(np.reshape(all_sampled_points_id, (all_sampled_points_id.shape[0], 1)))
    all_sampled_points_id = np.unique(all_sampled_points_id)
    tst_arr = np.zeros((all_sampled_points_id.shape[0], 2))
    all_sampled_points_id = pd.DataFrame(tst_arr, index=all_sampled_points_id).sort_index()

    del tst_arr

    # organizing sampled points for each image
    # setting up dataframes
    images_sampled_points = []
    image_names['check'] = np.zeros(image_names.shape[0])
    for i, (dat, ids) in enumerate(zip(data, image_ids)):
        if not image_names['check'].loc[ids[0]]:
            samples1 = all_sampled_points_id.copy()
            samples1[samples1.index.isin(dat.index)] = dat.iloc[:, :2].astype(float)  # .sort_index().drop_duplicates()

            images_sampled_points.append(samples1)
            # change to True - already computed those images
            image_names['check'].loc[ids[0]] = 1

    # extract appx ground points and group by mean value for points that were
    # computed more than once over multiple images
    all_appx_ground_points = pd.concat(data).iloc[:, 2:None].sort_index()
    all_appx_ground_points = all_appx_ground_points.groupby(all_appx_ground_points.index).mean()

    # fixing GCP for correct values (known, not computed)
    gc_mask = list(map(lambda x: x.startswith('C'), all_appx_ground_points.index.tolist()))
    all_appx_ground_points[gc_mask] = gcp.loc[all_appx_ground_points[gc_mask].index].astype(float)

    fig_orthographic = plt.figure()
    ax = fig_orthographic.add_subplot(111, projection='3d')
    plot_frames_and_points(cam, images, imageWidth, imageHeight, all_appx_ground_points, ax, scale_frame=1,
                           scale_axis=5)

    # starting adjustment
    res = non_linear_LSA(cam, images, all_appx_ground_points, images_sampled_points, epsilon=1e-4, iters=10, LM=False)

    # res = [rel_results, l_a, v, sigmaX, sig]
    calb_res = res[0]
    orig_samples = res[1]
    v = res[2]
    sigmaX = res[3]
    sig_apost = res[-1]

    # pretty print calibration results
    print_res(res, sigmaX, images)

    # print exceution time
    executionTime = (time.time() - startTime)
    print('Execution time in minutes: ' + str(executionTime / 60))

    pd.DataFrame(calb_res).to_csv('results.csv')
    pd.DataFrame(orig_samples).to_csv('orig_samples.csv')
    pd.DataFrame(sig_apost).to_csv('sig_apost.csv')
    pd.DataFrame(v).to_csv('v.csv')
    pd.DataFrame(sigmaX).to_csv('sigmaX.csv')


