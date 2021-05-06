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
from misc import *

pd.set_option('display.precision', 5)


def update_objects(cam, images, appx_ground_points, dX, num_cp):
    """
    updating objects and data for the adjustment process :)
    :param cam: updating f, principal point etc..
    :param images: updating EOP for every img
    :param appx_ground_points: updating every appx ground coordinate
    :return: none
    """
    # update EOP
    i = 0
    j = 6
    for img in images:
        img.exteriorOrientationParameters = np.add(img.exteriorOrientationParameters, np.reshape(dX[i:j], 6))
        i += 6
        j += 6

    # update cam params
    nidx = len(images) * 6  # times 6 eo parameters
    df = float(dX[nidx])
    dxp = float(dX[nidx + 1])
    dyp = float(dX[nidx + 2])
    dK1 = float(dX[nidx + 3])
    dK2 = float(dX[nidx + 4])
    dK3 = float(dX[nidx + 5])
    dP1 = float(dX[nidx + 6])
    dP2 = float(dX[nidx + 7])
    dB1 = float(dX[nidx + 8])
    dB2 = float(dX[nidx + 9])
    cam.focalLength = cam.focalLength + df
    cam.principalPoint[0] = cam.principalPoint[0] + dxp
    cam.principalPoint[1] = cam.principalPoint[1] + dyp
    cam.radialDistortions = np.add(cam.radialDistortions, np.array([dK1, dK2, dK3]))
    cam.decenteringDistortions = np.add(cam.decenteringDistortions, np.array([dP1, dP2]))
    cam.affinityDistortions = np.add(cam.affinityDistortions, np.array([dB1, dB2]))

    # update Tie Points
    dtp = dX[nidx + 10:None]

    appx_ground_points.iloc[:-num_cp, :] = np.reshape(
        np.add(appx_ground_points.iloc[:-num_cp, :].values.flatten()[:, None], dtp),
        appx_ground_points.iloc[:-num_cp, :].shape)


def non_linear_LSA(cam, images, all_appx_ground_points, images_sampled_points, epsilon=1e-6, iters=10, LM=False):
    #
    # # attempt to convert to sparse format
    # # sN = sparse.csr_matrix(N)
    # # sU = sparse.csr_matrix(u)
    # #
    # # dX = spsolve(sN, sU)[:, None]
    #######
    #########################################################################
    # try implementing LM algo
    #############################################################
    iter = 1
    dX = 999

    num_cp = all_appx_ground_points.loc[all_appx_ground_points.index.str.startswith('T')].shape[0]

    if LM:
        observs = []
        lamdas = []
        deltaxs = []
        lamda = 999
        while lamda > 1e-4:
            # compute new observation vector & design matrix
            ll0 = []
            ll1 = []
            for img, samp in zip(images, images_sampled_points):
                l0 = cam.ComputeObservationVector(img, np.array(all_appx_ground_points), np.array(samp))
                samp_temp = np.array(samp).flatten()
                l0[np.argwhere(samp_temp == 0)] = 0
                ll0.append(l0[:, None])
                ll1.append(samp_temp[:, None])

            L = (np.vstack(ll1) - np.vstack(ll0))  # observation vector
            L = L[L != 0][:, None]

            # computing design matrices and stacking
            aa_EOP = []
            aa_REST = []
            design_matrices = []
            # num_cp = all_sampled_points_id.loc[all_sampled_points_id.index.str.startswith('T')].shape[0]
            for i, img, samp in zip(range(len(images)), images, images_sampled_points):
                design_matrices.append(cam.ComputeDesignMatrix(img, np.array(all_appx_ground_points), np.array(samp)))
                design_matrices[-1] = design_matrices[-1][
                    np.reshape((ll0[i] != 0), ((ll0[i] != 0).shape[0]))]  # get only relevant rows
                aa_EOP.append(design_matrices[-1][:, 0:6])
                aa_REST.append(design_matrices[-1][:, 6:None])

            A = np.hstack((la.block_diag(*aa_EOP), np.vstack(aa_REST)[:, :-num_cp * 3]))

            # compute LM matrices
            N = np.dot(A.T, A)
            g = np.dot(A.T, L)
            # make sure lamda is created only once using this method
            if iter == 0:
                lamda = np.max(np.diag(N)) * 1e-6  # had to decrease this to a millionth of the max diagonal value
            iter += 1

            # checking if N+lamdaEYE is singular
            sing = 1
            while sing:
                H = N + lamda * np.eye(N.shape[0])
                try:
                    la.inv(H)
                except:
                    lamda *= 2
                    continue
                sing = 0

            # solving for dX
            dX_new = la.solve(H, g)

            # saving parameters to lists
            lamdas.append(lamda)
            observs.append(la.norm(L))
            deltaxs.append(la.norm(dX_new))

            # make copies of the arrays to not lose last iteration data
            temp_cam, temp_all_appx_ground_points, temp_images, temp_images_sampled_points = cam.copy(), all_appx_ground_points.copy(), images.copy(), images_sampled_points.copy()

            # check if conditions are met
            # update_adjustment_elements(temp_im_ground_points, temp_im_camera_points, temp_eop, dX_new)
            update_objects(temp_cam, temp_images, temp_all_appx_ground_points, dX_new, num_cp)

            # _, _, lb, l0 = compute_adjustment_matrices(temp_im_ground_points, temp_im_camera_points, temp_eop, f)
            # compute new observation vector & design matrix
            ll0 = []
            ll1 = []
            for img, samp in zip(temp_images, images_sampled_points):
                l0 = cam.ComputeObservationVector(img, np.array(all_appx_ground_points), np.array(samp))
                samp_temp = np.array(samp).flatten()
                l0[np.argwhere(samp_temp == 0)] = 0
                ll0.append(l0[:, None])
                ll1.append(samp_temp[:, None])

            # L = (np.vstack(ll1) - np.vstack(ll0))  # observation vector
            # L = L[L != 0][:, None]

            check = la.norm(np.vstack(ll1) - np.vstack(ll0))  # edit this to not include zeros
            if la.norm(np.vstack(ll1) - np.vstack(ll0)) < la.norm(L):  # la.norm(dX_new) < la.norm(dX) and
                # if conditions are met we want to decrease lamda
                lamda /= 3
                dX = dX_new
                # update_adjustment_elements(im_ground_points, im_camera_points, eop, dX)
                update_objects(cam, images, all_appx_ground_points, dX, num_cp)
            else:
                # otherwise we want to increase lamda - alternating between gradient-descent and gauss-newton
                lamda *= 2  # LM not yet implemented

    # defining table for mid results
    mid_res_table = PrettyTable()
    mid_res_table.field_names = ['iter', 'v.T * v', '||dx_camera||', '||dx_tie_points||']

    while la.norm(dX) > epsilon and iter < iters:
        # compute new observation vector & design matrix
        ll0 = []
        ll1 = []
        for img, samp in zip(images, images_sampled_points):
            l0 = cam.ComputeObservationVector(img, np.array(all_appx_ground_points), np.array(samp))
            samp_temp = np.array(samp).flatten()
            l0[np.argwhere(samp_temp == 0)] = 0
            ll0.append(l0[:, None])
            ll1.append(samp_temp[:, None])

        L = (np.vstack(ll1) - np.vstack(ll0))  # observation vector
        L = L[L != 0][:, None]

        # computing design matrices and stacking
        aa_EOP = []
        aa_REST = []
        design_matrices = []
        # num_cp = all_sampled_points_id.loc[all_sampled_points_id.index.str.startswith('T')].shape[0]
        for i, img, samp in zip(range(len(images)), images, images_sampled_points):
            design_matrices.append(cam.ComputeDesignMatrix(img, np.array(all_appx_ground_points), np.array(samp)))
            design_matrices[-1] = design_matrices[-1][
                np.reshape((ll0[i] != 0), ((ll0[i] != 0).shape[0]))]  # get only relevant rows
            aa_EOP.append(design_matrices[-1][:, 0:6])
            aa_REST.append(design_matrices[-1][:, 6:None])

        A = np.hstack((la.block_diag(*aa_EOP), np.vstack(aa_REST)[:, :-num_cp * 3]))  # [:, :-num_cp]

        # compute next iterations
        N = np.dot(A.T, A)
        u = np.dot(A.T, L)

        # attempt to convert to sparse format
        # sN = sparse.csr_matrix(N)
        # sU = sparse.csr_matrix(u)
        #
        # dX = spsolve(sN, sU)[:, None]
        ######

        dX = la.solve(N, u)

        # update orientation pars
        update_objects(cam, images, all_appx_ground_points, dX, num_cp)
        norm = la.norm(dX)
        mid_res = pd.DataFrame(
            [float(cam.focalLength), float(cam.principalPoint[0]), float(cam.principalPoint[1]),
             1e-5 * cam.radialDistortions[0],
             1e-10 * cam.radialDistortions[1], 1e-12 * cam.radialDistortions[2], 1e-5 * cam.decenteringDistortions[0],
             1e-5 * cam.decenteringDistortions[1], 1e-5 * cam.affinityDistortions[0],
             1e-5 * cam.affinityDistortions[1]],
            index=['f', 'xp', 'yp', 'K1', 'K2', 'K3', 'P1', 'P2', 'B1', 'B2'])
        # print('norm: ', la.norm(dX))
        nidx = len(images) * 6
        dx_camera = la.norm(dX[nidx:nidx + 10])
        dx_tie_points = la.norm(dX[nidx + 10:None])
        # print(mid_res)
        sig_apost = np.dot(L.T, L)

        '{0:.2f}'.format(float(sig_apost))
        mid_res_table.add_row(
            [iter, '{0:.5f}'.format(float(sig_apost)), '{0:.5E}'.format(dx_camera), '{0:.5E}'.format(dx_tie_points)])

        iter += 1

    # print mid results for iterations
    print(mid_res_table)

    # compute residuals
    l_a = []
    for img, samp in zip(images, images_sampled_points):
        l0 = cam.ComputeObservationVector(img, np.array(all_appx_ground_points), np.array(samp))
        samp_temp = np.array(samp).flatten()
        l0[np.argwhere(samp_temp == 0)] = 0
        l_a.append(l0[:, None])

    v = np.vstack(l_a) - np.vstack(ll1)
    v = v[v != 0][:, None]  # clear zeros
    if (np.size(A, 0) - np.size(dX)) != 0:
        sig = np.dot(v.T, v) / (np.size(A, 0) - np.size(dX))
        sigmaX = sig[0] * la.inv(N)
    else:
        sigmaX = None

    rel_results = pd.DataFrame(
        [iter, la.norm(dX), float(cam.focalLength), float(cam.principalPoint[0]), float(cam.principalPoint[1]),
         1e-5 * cam.radialDistortions[0],
         1e-10 * cam.radialDistortions[1], 1e-12 * cam.radialDistortions[2], 1e-5 * cam.decenteringDistortions[0],
         1e-5 * cam.decenteringDistortions[1], 1e-5 * cam.affinityDistortions[0], 1e-5 * cam.affinityDistortions[1]],
        index=['iter', 'norm', 'f', 'xp', 'yp', 'K1', 'K2', 'K3', 'P1', 'P2', 'B1', 'B2'])

    # sorting out the original sampled points to return-
    ll1 = np.vstack(ll1)
    ll1 = ll1[ll1 != 0][:, None]

    return [rel_results, ll1, v, sigmaX, sig]


if __name__ == '__main__':
    startTime = time.time()
    # setting initial camera parameters
    f = 50  # mm
    xp = 0  # m
    yp = 0  # m
    sensor_size = 5.2e-3  # mm
    meter_per_pixel = 0.05  # m
    imageWidth_pix, imageHeight_pix = 10328., 7760.
    imageWidth, imageHeight = imageWidth_pix * meter_per_pixel, imageHeight_pix * meter_per_pixel

    eop_path = r"raw_data\med_format\eop_full_block.txt"
    points_path = r"raw_data\med_format\cpoints\*.txt"

    # creating camera object
    a = imageWidth_pix * sensor_size / 2
    b = imageHeight_pix * sensor_size / 2
    fiducials = np.array([[-a, -b], [a, b], [-a, b], [a, -b]])
    # define 'fiducials' for images
    # could be added via txt file later or from img size if available
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
            data.append(np.vstack(cpoints_imgs))
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

    # convert image samples into camera system and to ground system
    appx_ground_data = []
    for i, dat in enumerate(data):
        # converting to float and adjusting dataframe
        data[i] = pd.DataFrame(dat[:, 1:5], index=dat[:, 0]).astype(float)
        # every image object holds points sampled in a two image model
        img0id = cp_file_names[i].split('+')[0]
        img1id = cp_file_names[i].split('+')[1]
        data[i].loc[:, 0:1] = images[int(image_names.loc[img0id])].ImageToCamera(data[i].loc[:, 0:1])
        data[i].loc[:, 2:3] = images[int(image_names.loc[img1id])].ImageToCamera(data[i].loc[:, 2:3])
        # get points in ground system by geometric intersection
        appx_ground_data = ImagesToGround(images[int(image_names.loc[img0id])],
                                          images[int(image_names.loc[img1id])],
                                          data[i].loc[:, 0:1], data[i].loc[:, 2:3])
        appx_ground_data = pd.DataFrame(appx_ground_data, index=data[i].index)
        data[i] = pd.concat([data[i], appx_ground_data], axis=1)
        data[i].columns = ['x1[mm]', 'y1[mm]', 'x2[mm]', 'y2[mm]', 'X[m]', 'Y[m]', 'Z[m]']

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
        if not image_names['check'].loc[ids[0]] and not image_names['check'].loc[ids[1]]:
            samples1 = all_sampled_points_id.copy()
            samples2 = all_sampled_points_id.copy()
            samples1[samples1.index.isin(dat.index)] = dat.iloc[:, 0:2].astype(float)
            samples2[samples2.index.isin(dat.index)] = dat.iloc[:, 2:4].astype(float)

            images_sampled_points.append(samples1)
            images_sampled_points.append(samples2)

            # change to True - already computed those images
            image_names['check'].loc[ids[0]] = 1
            image_names['check'].loc[ids[1]] = 1  ### WARNING ###

        elif image_names['check'].loc[ids[0]] and not image_names['check'].loc[ids[1]]:
            # samples1 = all_sampled_points_id.copy()
            samples2 = all_sampled_points_id.copy()
            images_sampled_points[-1][images_sampled_points[-1].index.isin(dat.index)] = dat.iloc[:, 0:2].astype(
                float)
            samples2[samples2.index.isin(dat.index)] = dat.iloc[:, 2:4].astype(float)

            images_sampled_points.append(samples2)

            # change to True - already computed those images
            image_names['check'].loc[ids[1]] = 1

        elif not image_names['check'].loc[ids[0]] and image_names['check'].loc[ids[1]]:
            samples1 = all_sampled_points_id.copy()
            # samples2 = all_sampled_points_id.copy()
            samples1[samples1.index.isin(dat.index)] = dat.iloc[:, 0:2].astype(float)
            # samples2[samples2.index.isin(dat.index)] = dat.iloc[:, 2:4].astype(float)

            images_sampled_points.append(samples1)

            # change to True - already computed those images
            image_names['check'].loc[ids[0]] = 1

    # extract appx ground points and group by mean value for points that were
    # computed more than once over multiple images
    all_appx_ground_points = pd.concat(data).iloc[:, 4:None]
    all_appx_ground_points = all_appx_ground_points.groupby(all_appx_ground_points.index).mean()

    # fixing GCP for correct values (known, not computed)
    gc_mask = list(map(lambda x: x.startswith('T'), all_appx_ground_points.index.tolist()))
    all_appx_ground_points[gc_mask] = gcp.loc[all_appx_ground_points[gc_mask].index].astype(float)

    """
    wrote the update object function. now we have to start the adjustment process and 
    hope it works and it comes to a convergence.
    after that the sky is the limit ?
    """
    fig_orthographic = plt.figure()
    ax = fig_orthographic.add_subplot(111, projection='3d')
    plot_frames_and_points(cam, images, imageWidth, imageHeight, all_appx_ground_points, ax, scale_frame=1,
                           scale_axis=5)

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
