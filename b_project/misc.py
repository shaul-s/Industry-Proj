from scipy import linalg as la
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from PhotoViewer import *


def ImagesToGround(image1, image2, imagePoints1, imagePoints2):
    """
    Computes ground coordinates of homological points
    :param imagePoints1: points in image 1
    :param imagePoints2: corresponding points in image 2
    :param Method: method to use for the ray intersection, three options exist: geometric, vector, Collinearity

    :type imagePoints1: np.array nx2
    :type imagePoints2: np.array nx2
    :type Method: string

    :return: ground points, their accuracies.
    :rtype: dict
    """
    #  defining perspective center in the world system and transforming to camera points
    o1 = np.array(image1.exteriorOrientationParameters[0:3])
    o2 = np.array(image2.exteriorOrientationParameters[0:3])
    # camPoints1 = self.__image1.ImageToCamera(imagePoints1)
    # camPoints2 = self.__image2.ImageToCamera(imagePoints2)
    camPoints1 = imagePoints1.values
    camPoints2 = imagePoints2.values

    x1 = np.hstack((camPoints1, np.full((camPoints1.shape[0], 1), -image1.camera.focalLength)))
    x2 = np.hstack((camPoints2, np.full((camPoints2.shape[0], 1), -image2.camera.focalLength)))

    v1 = np.dot(image1.rotationMatrix, x1.T).T  #
    v1 = v1 / la.norm(v1, axis=1, keepdims=True)
    v2 = np.dot(image2.rotationMatrix, x2.T).T  #
    v2 = v2 / la.norm(v2, axis=1, keepdims=True)

    # getting the adjustment matrices ready
    aa = []
    ll = []
    for vec1, vec2 in zip(v1, v2):
        i_v1vt = np.eye(vec1.shape[0]) - np.dot(vec1[:, None], vec1[:, None].T)
        i_v2vt = np.eye(vec2.shape[0]) - np.dot(vec2[:, None], vec2[:, None].T)
        aa.append(np.vstack((i_v1vt, i_v2vt)))
        ll.append(np.vstack((np.dot(i_v1vt, o1)[:, None], np.dot(i_v2vt, o2)[:, None])))

    A = la.block_diag(*aa)
    L = np.vstack(ll)

    # LS Adjustment geometric intersection
    N = np.dot(A.T, A)
    u = np.dot(A.T, L)

    X = np.reshape(la.solve(N, u), x1.shape)

    # compute accuracies in avg distance of ray to optimal pnt
    accuracies = []
    for x, a in zip(X, aa):
        e1 = np.dot(a[0:3, :], x - o1)
        e2 = np.dot(a[3:None, :], x - o2)
        accuracies.append(la.norm((np.abs(e1) + np.abs(e2)) / 2))

    return X


def ImageToGround_GivenZ(cameraPoints, Z_values, eop, f):
    """
    Compute corresponding ground point given the height in world system

    :param cameraPoints: points in camera space
    :param Z_values: height of the ground points
    :param eop: exterior orientation parameters
    :param f: focal length

    :type Z_values: np.array nx1
    :type cameraPoints: np.array nx2
    :type eop: np.array 6x1
    :type f: float

    :return: corresponding ground points

    :rtype: np.ndarray
    """
    X0 = eop[0]
    Y0 = eop[1]
    Z0 = eop[2]

    T = np.array([[X0], [Y0], [Z0]])

    omega = eop[3]
    phi = eop[4]
    kappa = eop[5]
    R = Compute3DRotationMatrix(omega, phi, kappa)

    # allocating memory for return array
    groundPoints = []

    for i in range(cameraPoints.shape[0]):
        camVec = cameraPoints[i, :]
        lam = float((Z_values[i] - Z0) / (np.dot(R[2, :], camVec)))

        X = X0 + lam * np.dot(R[0, :], camVec)
        Y = Y0 + lam * np.dot(R[1, :], camVec)

        xy = [X, Y, float(Z_values[i])]
        groundPoints.append(xy)

    groundPoints = np.array(groundPoints)

    return groundPoints


def print_res(res, sigmaX, images):
    """
    pretty printing camera calibration results
    """
    std = np.sqrt(np.diag(sigmaX)[len(images) * 6:len(images) * 6 + 10])
    std[3] = 1e-5 * std[3]
    std[4] = 1e-10 * std[4]
    std[5] = 1e-12 * std[5]
    std[6] = 1e-5 * std[6]
    std[7] = 1e-5 * std[7]
    std[8] = 1e-5 * std[8]
    std[9] = 1e-5 * std[9]
    res_table = PrettyTable()
    res_table.field_names = ['IO Parameters', 'Value', 'STD']
    res_table.add_row(['Focal Length', '{0:.4f}(mm)'.format(res[0].loc['f'][0]), u'\u00B1{0:.4f}(mm)'.format(std[0])])
    res_table.add_row(['PRINCIPAL POINT OFFSET', '-', '-'])
    res_table.add_row(['XP', '{0:.4f}(mm)'.format(res[0].loc['xp'][0]), u'\u00B1{0:.4f}(mm)'.format(std[1])])
    res_table.add_row(['YP', '{0:.4f}(mm)'.format(res[0].loc['yp'][0]), u'\u00B1{0:.4f}(mm)'.format(std[2])])
    res_table.add_row(['RADIAL DISTORTIONS', '-', '-'])
    res_table.add_row(['K1', '{0:.5E}'.format(res[0].loc['K1'][0]), u'\u00B1{0:.5E}'.format(std[3])])
    res_table.add_row(['K2', '{0:.5E}'.format(res[0].loc['K2'][0]), u'\u00B1{0:.5E}'.format(std[4])])
    res_table.add_row(['K3', '{0:.5E}'.format(res[0].loc['K3'][0]), u'\u00B1{0:.5E}'.format(std[5])])
    res_table.add_row(['TANGENTIAL DISTORTIONS', '-', '-'])
    res_table.add_row(['P1', '{0:.5E}'.format(res[0].loc['P1'][0]), u'\u00B1{0:.5E}'.format(std[6])])
    res_table.add_row(['P2', '{0:.5E}'.format(res[0].loc['P2'][0]), u'\u00B1{0:.5E}'.format(std[7])])
    res_table.add_row(['AFFINITY AND NON-ORTHOGONALITY', '-', '-'])
    res_table.add_row(['B1', '{0:.5E}'.format(res[0].loc['B1'][0]), u'\u00B1{0:.5E}'.format(std[8])])
    res_table.add_row(['B2', '{0:.5E}'.format(res[0].loc['B2'][0]), u'\u00B1{0:.5E}'.format(std[9])])

    print(res_table)


def plot_frames_and_points(cam, images, imageWidth, imageHeight, ground_points, ax, scale_frame=1, scale_axis=1):
    """
    plots the image frames, ground control points and tie points
    in ground system 3d axis
    """
    f = cam.focalLength / 1000
    for img in images:
        z = 37.37
        drawImageFrame(imageWidth, imageHeight, img.rotationMatrix,
                       np.vstack((img.exteriorOrientationParameters[:2, None], [z])),
                       f, scale_frame, ax)

        drawOrientation(img.rotationMatrix, img.exteriorOrientationParameters[:3, None], scale_axis, ax)

    tie_mask = list(map(lambda x: x[0].isdigit(), ground_points.index.tolist()))
    gc_mask = list(map(lambda x: x.startswith('C'), ground_points.index.tolist()))

    tie_points = ground_points[tie_mask].values.astype(float)
    gcp = ground_points[gc_mask].values.astype(float)

    ax.scatter(gcp[:, 0], gcp[:, 1], gcp[:, 2], c='k', s=100, marker='^', label='control point')
    ax.scatter(tie_points[:, 0], tie_points[:, 1], tie_points[:, 2], c='c', s=0.75, marker='^', label='tie point')
    ax.legend()
    plt.show()


def non_linear_LSA(cam, images, all_appx_ground_points, images_sampled_points, epsilon=1e-6, iters=10, LM=False):
    #
    # # attempt to convert to sparse format
    # # sN = sparse.csr_matrix(N)
    # # sU = sparse.csr_matrix(u)
    # #
    # # dX = spsolve(sN, sU)[:, None]

    #############################################################
    # try implementing LM algo
    #############################################################
    iter = 1
    dX = 999

    num_cp = all_appx_ground_points.loc[all_appx_ground_points.index.str.startswith('C')].shape[0]

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
