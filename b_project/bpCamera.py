import numpy as np
from scipy import linalg as la
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix


class Camera(object):

    def __init__(self, focal_length, principal_point, fiducial_marks, sensor_size):
        """
        Initialize the Camera object

        :param focal_length: focal length of the camera(mm)
        :param principal_point: principle point 1x2 (mm)
        :param radial_distortions: the radial distortion parameters K0, K1, K2 ...
        :param decentering_distortions: decentering distortion parameters P0, P1, P2 ...
        :param fiducial_marks: for inner orientation
        :param sensor_size: size of sensor

        :type focal_length: double
        :type principal_point: np.array
        :type radial_distortions: np.array
        :type decentering_distortions: np.array
        :type fiducial_marks: np.array
        :type sensor_size: double

        """
        # private parameters
        self.__focal_length = focal_length
        self.__principal_point = principal_point
        self.__radial_distortions = np.zeros(3)
        self.__decentering_distortions = np.zeros(2)
        self.__affinity_distortions = np.zeros(2)
        self.__fiducial_marks = fiducial_marks
        self.__CalibrationParam = np.zeros(10)  # 10 calibration params
        self.__sensor_size = sensor_size

    def __copy__(self):
        # self.__focal_length = focal_length
        #         self.__principal_point = principal_point
        #         self.__radial_distortions = np.zeros(3)
        #         self.__decentering_distortions = np.zeros(2)
        #         self.__affinity_distortions = np.zeros(2)
        #         self.__fiducial_marks = fiducial_marks
        #         self.__CalibrationParam = np.zeros(10)  # 10 calibration params
        #         self.__sensor_size = sensor_size
        return type(self)(self.__focal_length.self.__principal_point, self.__radial_distortions,
                      self.__decentering_distortions,
                      self.__affinity_distortions, self.__fiducial_marks, self.__CalibrationParam, self.__sensor_size)

    @property
    def focalLength(self):
        """
        Focal length of the camera
        :return: focal length
        :rtype: float
        """
        return self.__focal_length

    @focalLength.setter
    def focalLength(self, val):
        """
        Set the focal length value
        :param val: value for setting
        :type: float
        """
        self.__focal_length = val

    @property
    def principalPoint(self):
        """
        Principal point of the camera
        :return: principal point coordinates
        :rtype: np.ndarray
        """
        return self.__principal_point

    @principalPoint.setter
    def principalPoint(self, val):
        """
        Set the principal point value
        :param val: value for setting
        :type: float
        """

        self.self.__principal_point = val

    @property
    def radialDistortions(self):
        """
        radial distortions K1 and K2
        :return: radial distortions K1 and K2
        :rtype: dict
        """
        return self.__radial_distortions

    @radialDistortions.setter
    def radialDistortions(self, value):
        self.__radial_distortions = value

    @property
    def decenteringDistortions(self):
        """
        decentring distortions K1 and K2
        :return: decentring distortions K1 and K2
        :rtype: dict
        """
        return self.__decentering_distortions

    @decenteringDistortions.setter
    def decenteringDistortions(self, value):
        """

        :param value:
        :return:
        """
        self.__decentering_distortions = value

    @property
    def affinityDistortions(self):
        """
        affinity distortions B1 and B2
        :return: decentring distortions K1 and K2
        :rtype: dict
        """
        return self.__affinity_distortions

    @affinityDistortions.setter
    def affinityDistortions(self, value):
        """

        :param value:
        :return:
        """
        self.__affinity_distortions = value

    @property
    def sensorSize(self):
        """
        Sensor size of the camera
        :return: sensor size
        :rtype: float
        """
        return self.__sensor_size

    @property
    def CalibrationParam(self):
        """
        returns calib params array
        """
        return self.__CalibrationParam

    @CalibrationParam.setter
    def CalibrationParam(self, parametersArray):
        """
        :param parametersArray: the parameters to update the ``self.__CalibrationParam``
        """
        self.__CalibrationParam = parametersArray

    @property
    def fiducials(self):
        """
        returns the fiducials of the camera
        """
        return self.__fiducial_marks

    @fiducials.setter
    def fiducials(self, parametersArray):
        """
        :param parametersArray: the parameters to update the ``self.__fiducial_marks``
        """
        self.__fiducial_marks = parametersArray

    # ---------------------- Private methods ---------------------- #

    def ComputeObservationVector(self, img, groundPoints, cameraPoints):
        """
        Compute observation vector for the adjustment process
        atm using k1, k2
        :param img:
        :param groundPoints:
        :param cameraPoints:
        :return:
        """
        # initialization for readability
        xp = self.principalPoint[0]
        yp = self.principalPoint[1]
        f = self.focalLength
        k1 = self.radialDistortions[0]
        k2 = self.radialDistortions[1]
        k3 = self.radialDistortions[2]
        p1 = self.decenteringDistortions[0]
        p2 = self.decenteringDistortions[1]
        b1 = self.affinityDistortions[0]
        b2 = self.affinityDistortions[1]

        n = groundPoints.shape[0]  # number of points

        # Coordinates subtraction
        dX = groundPoints[:, 0] - float(img.exteriorOrientationParameters[0])
        dY = groundPoints[:, 1] - float(img.exteriorOrientationParameters[1])
        dZ = groundPoints[:, 2] - float(img.exteriorOrientationParameters[2])
        dXYZ = np.vstack([dX, dY, dZ])
        rotated_XYZ = np.dot(img.rotationMatrix.T, dXYZ).T

        range = (cameraPoints[:, 1] - yp) ** 2 + (cameraPoints[:, 0] - xp) ** 2

        # l0 initialization
        l0 = np.empty(n * 2)

        # Computation of the observation vector based on approximate EOP & IOP:
        l0[::2] = xp - f * rotated_XYZ[:, 0] / rotated_XYZ[:, 2] - (
                cameraPoints[:, 0] - xp) * (
                          1e-5 * k1 * range + 1e-10 * k2 * (range ** 2) + 1e-12 * k3 * (
                          range ** 3)) - 1e-5 * p1 * (
                          range + 2 * (cameraPoints[:, 0] - xp) ** 2) - 2 * 1e-5 * p2 * (
                          cameraPoints[:, 0] - xp) * (
                          cameraPoints[:, 1] - yp) - 1e-5 * b1 * (cameraPoints[:, 0] - xp) - 1e-5 * b2 * (
                          cameraPoints[:, 1] - yp)
        l0[1::2] = yp - f * rotated_XYZ[:, 1] / rotated_XYZ[:, 2] - (
                cameraPoints[:, 1] - yp) * (1e-5 * k1 * range + 1e-10 * k2 * (range ** 2) + 1e-12 * k3 * (
                range ** 3)) - 2 * 1e-5 * p1 * (cameraPoints[:, 0] - xp) * (cameraPoints[:, 1] - yp) - 1e-5 * p2 * (
                           range + 2 * (cameraPoints[:, 1] - yp) ** 2)

        return l0

    def ComputeDesignMatrix(self, img, groundPoints, cameraPoints):
        """
        Compute the derivatives of the collinear law (design matrix)
        :param: img: the image in a singleimage object
        :param groundPoints: Ground coordinates of every point
        :param cameraPoints: Camera coordinates of every point
        :type img: bpSingleImage
        :type groundPoints: np.array nx3
        :type cameraPoints: nparray nx2
        :return: The design matrix
        :rtype: np.array nxUNKNOWNS
        """
        # initialization for readability
        xp = self.principalPoint[0]
        yp = self.principalPoint[1]
        f = self.focalLength
        k1 = self.radialDistortions[0]
        k2 = self.radialDistortions[1]
        k3 = self.radialDistortions[2]
        p1 = self.decenteringDistortions[0]
        p2 = self.decenteringDistortions[1]
        b1 = self.affinityDistortions[0]
        b2 = self.affinityDistortions[1]
        omega = img.exteriorOrientationParameters[3]
        phi = img.exteriorOrientationParameters[4]
        kappa = img.exteriorOrientationParameters[5]

        calib_unknowns = [f, xp, yp, k1, k2, k3, p1, p2, b1, b2]

        range = (cameraPoints[:, 1] - yp) ** 2 + (cameraPoints[:, 0] - xp) ** 2

        # Coordinates subtraction
        dX = groundPoints[:, 0] - img.exteriorOrientationParameters[0]
        dY = groundPoints[:, 1] - img.exteriorOrientationParameters[1]
        dZ = groundPoints[:, 2] - img.exteriorOrientationParameters[2]
        dXYZ = np.vstack([dX, dY, dZ])

        rotationMatrixT = img.rotationMatrix.T
        rotatedG = rotationMatrixT.dot(dXYZ)
        rT1g = rotatedG[0, :]
        rT2g = rotatedG[1, :]
        rT3g = rotatedG[2, :]

        focalBySqauredRT3g = f / rT3g ** 2

        dxdg = rotationMatrixT[0, :][None, :] * rT3g[:, None] - rT1g[:, None] * rotationMatrixT[2, :][None, :]
        dydg = rotationMatrixT[1, :][None, :] * rT3g[:, None] - rT2g[:, None] * rotationMatrixT[2, :][None, :]

        dgdX0 = np.array([-1, 0, 0], 'f')
        dgdY0 = np.array([0, -1, 0], 'f')
        dgdZ0 = np.array([0, 0, -1], 'f')
        dgdXj = np.array([1, 0, 0], 'f')
        dgdYj = np.array([0, 1, 0], 'f')
        dgdZj = np.array([0, 0, 1], 'f')

        # Derivatives with respect to X0
        dxdX0 = -focalBySqauredRT3g * np.dot(dxdg, dgdX0)
        dydX0 = -focalBySqauredRT3g * np.dot(dydg, dgdX0)

        # Derivatives with respect to Y0
        dxdY0 = -focalBySqauredRT3g * np.dot(dxdg, dgdY0)
        dydY0 = -focalBySqauredRT3g * np.dot(dydg, dgdY0)

        # Derivatives with respect to Z0
        dxdZ0 = -focalBySqauredRT3g * np.dot(dxdg, dgdZ0)
        dydZ0 = -focalBySqauredRT3g * np.dot(dydg, dgdZ0)

        # Derivatives with respect to Xj
        dxdXj = -focalBySqauredRT3g * np.dot(dxdg, dgdXj)
        dxdXj[cameraPoints[:, 0] == 0] = 0
        # dxdXj = dxdXj[:-num_cp]
        dydXj = -focalBySqauredRT3g * np.dot(dydg, dgdXj)
        dydXj[cameraPoints[:, 1] == 0] = 0
        # dydXj = dydXj[:-num_cp]

        # Derivatives with respect to Yj
        dxdYj = -focalBySqauredRT3g * np.dot(dxdg, dgdYj)
        dxdYj[cameraPoints[:, 0] == 0] = 0
        # dxdYj = dxdYj[:-num_cp]
        dydYj = -focalBySqauredRT3g * np.dot(dydg, dgdYj)
        dydYj[cameraPoints[:, 1] == 0] = 0
        # dydYj = dydYj[:-num_cp]

        # Derivatives with respect to Zj
        dxdZj = -focalBySqauredRT3g * np.dot(dxdg, dgdZj)
        dxdZj[cameraPoints[:, 0] == 0] = 0
        # dxdZj = dxdZj[:-num_cp]
        dydZj = -focalBySqauredRT3g * np.dot(dydg, dgdZj)
        dydZj[cameraPoints[:, 1] == 0] = 0
        # dydZj = dydZj[:-num_cp]

        # All tie points derivatives organized
        tpd = np.zeros((2 * len(dxdXj), 3 * len(dxdXj)))

        dxXYZj = np.hstack((dxdXj[:, None], dxdYj[:, None], dxdZj[:, None]))
        dyXYZj = np.hstack((dydXj[:, None], dydYj[:, None], dydZj[:, None]))

        tpd[0::2] = la.block_diag(*dxXYZj)
        tpd[1::2] = la.block_diag(*dyXYZj)

        # Derivatives with respect to rotation angles
        dRTdOmega = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'omega').T
        dRTdPhi = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'phi').T
        dRTdKappa = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'kappa').T

        gRT3g = dXYZ * rT3g

        # Derivatives with respect to Omega
        dxdOmega = -focalBySqauredRT3g * (dRTdOmega[0, :][None, :].dot(gRT3g) -
                                          rT1g * (dRTdOmega[2, :][None, :].dot(dXYZ)))[0]

        dydOmega = -focalBySqauredRT3g * (dRTdOmega[1, :][None, :].dot(gRT3g) -
                                          rT2g * (dRTdOmega[2, :][None, :].dot(dXYZ)))[0]

        # Derivatives with respect to Phi
        dxdPhi = -focalBySqauredRT3g * (dRTdPhi[0, :][None, :].dot(gRT3g) -
                                        rT1g * (dRTdPhi[2, :][None, :].dot(dXYZ)))[0]

        dydPhi = -focalBySqauredRT3g * (dRTdPhi[1, :][None, :].dot(gRT3g) -
                                        rT2g * (dRTdPhi[2, :][None, :].dot(dXYZ)))[0]

        # Derivatives with respect to Kappa
        dxdKappa = -focalBySqauredRT3g * (dRTdKappa[0, :][None, :].dot(gRT3g) -
                                          rT1g * (dRTdKappa[2, :][None, :].dot(dXYZ)))[0]

        dydKappa = -focalBySqauredRT3g * (dRTdKappa[1, :][None, :].dot(gRT3g) -
                                          rT2g * (dRTdKappa[2, :][None, :].dot(dXYZ)))[0]

        # Derivatives with respect to f
        dxdf = -rT1g / rT3g
        dydf = -rT2g / rT3g

        # Derivatives with respect to xp
        dxdxp = np.full((cameraPoints.shape[0]), 1)
        dxdyp = np.full((cameraPoints.shape[0]), 0)

        # Derivatives with respect to yp
        dydxp = np.full((cameraPoints.shape[0]), 0)
        dydyp = np.full((cameraPoints.shape[0]), 1)

        # Derivatives with respect to K1
        dxdK1 = -1e-5 * (cameraPoints[:, 0] - xp) * range
        dydK1 = -1e-5 * (cameraPoints[:, 1] - yp) * range

        # Derivatives with respect to K2
        dxdK2 = -1e-10 * (cameraPoints[:, 0] - xp) * (range ** 2)
        dydK2 = -1e-10 * (cameraPoints[:, 1] - yp) * (range ** 2)

        # Derivatives with respect to K3
        dxdK3 = -1e-12 * (cameraPoints[:, 0] - xp) * (range ** 3)
        dydK3 = -1e-12 * (cameraPoints[:, 1] - yp) * (range ** 3)

        # Derivatives with respect to P1
        dxdP1 = -1e-5 * (2 * (cameraPoints[:, 0] - xp) ** 2 + range)
        dydP1 = -1e-5 * (2 * (cameraPoints[:, 0] - xp) * (cameraPoints[:, 1] - yp))

        # Derivatives with respect to P2
        dxdP2 = -1e-5 * (2 * (cameraPoints[:, 0] - xp) * (cameraPoints[:, 1] - yp))
        dydP2 = -1e-5 * (2 * (cameraPoints[:, 1] - yp) ** 2 + range)

        # Derivatives with respect to B1
        dxdB1 = -1e-5 * (cameraPoints[:, 0] - xp)
        dydB1 = np.zeros(cameraPoints.shape[0])

        # Derivatives with respect to B2
        dxdB2 = -1e-5 * (cameraPoints[:, 1] - yp)
        dydB2 = np.zeros(cameraPoints.shape[0])

        # all derivatives of x and y
        dd = np.array(
            [np.vstack([dxdX0, dxdY0, dxdZ0, dxdOmega, dxdPhi, dxdKappa, dxdf, dxdxp, dxdyp, dxdK1, dxdK2, dxdK3, dxdP1,
                        dxdP2, dxdB1, dxdB2]).T,
             np.vstack(
                 [dydX0, dydY0, dydZ0, dydOmega, dydPhi, dydKappa, dydf, dydxp, dydyp, dydK1, dydK2, dydK3, dydP1,
                  dydP2, dydB1, dydB2]).T])

        a = np.zeros((2 * dd[0].shape[0], 6 + len(calib_unknowns)))
        a[0::2] = dd[0]
        a[1::2] = dd[1]

        return np.hstack((a, tpd))


if __name__ == '__main__':
    pass
    # focal_length, principal_point, radial_distortions, decentering_distortions,
    # sensor_size)
    # cam = Camera(50, [0.1, 0.2], np.zeros(3), np.zeros(2), 1e-6)

    # print(cam)
