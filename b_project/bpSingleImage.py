import numpy as np
from scipy import linalg as la
from bpCamera import *
from MatrixMethods import *


class SingleImage(object):

    def __init__(self, id, camera):
        """
        Initialize the SingleImage object
        :param camera: instance of the Camera class

        :type camera: Camera

        """
        self.__id = id
        self.__camera = camera
        self.__sampled_pts = None
        self.__innerOrientationParameters = None
        self.__exteriorOrientationParameters = np.array([0, 0, 0, 0, 0, 0], 'f')
        self.__rotationMatrix = None

    @property
    def id(self):
        """
        The image id
        :rtype: str
        """
        return self.__id

    @property
    def camera(self):
        """
        The camera that took the image
        :rtype: Camera

        """
        return self.__camera

    @property
    def sampledPts(self):
        """
        sampled points in image space
        :rtype: np.array

        """
        return self.__sampled_pts

    @sampledPts.setter
    def sampledPts(self, val):
        """
        sampled points in image space
        :rtype: np.array

        """
        self.__sampled_pts = val

    @property
    def innerOrientationParameters(self):
        """
        Inner orientation parameters
        :return: inner orinetation parameters
        """
        return self.__innerOrientationParameters

    @innerOrientationParameters.setter
    def innerOrientationParameters(self, parametersArray):
        """
        :param parametersArray: the parameters to update the ``self.__innerOrientationParameters``
        """
        self.__innerOrientationParameters = parametersArray

    @property
    def exteriorOrientationParameters(self):
        r"""
        Property for the exterior orientation parameters

        :return: exterior orientation parameters in the following order, **however you can decide how to hold them (dictionary or array)**

        .. math::
            exteriorOrientationParameters = \begin{bmatrix} X_0 \\ Y_0 \\ Z_0 \\ \omega \\ \varphi \\ \kappa \end{bmatrix}

        :rtype: np.ndarray or dict
        """
        return self.__exteriorOrientationParameters

    @exteriorOrientationParameters.setter
    def exteriorOrientationParameters(self, parametersArray):
        """
        :param parametersArray: the parameters to update the ``self.__exteriorOrientationParameters``
        """
        self.__exteriorOrientationParameters = parametersArray

    @property
    def rotationMatrix(self):
        """
        The rotation matrix of the image

        Relates to the exterior orientation
        :return: rotation matrix

        :rtype: np.ndarray (3x3)
        """

        R = Compute3DRotationMatrix(self.exteriorOrientationParameters[3], self.exteriorOrientationParameters[4],
                                    self.exteriorOrientationParameters[5])

        return R

    def ComputeInnerOrientation(self, imagePoints):
        """
        Compute inner orientation parameters

        :param imagePoints: coordinates in image space

        :type imagePoints: np.array nx2

        :return: Inner orientation parameters, their accuracies, and the residuals vector

        :rtype: none
        """
        #  implementing observation vectors
        imagePoints = imagePoints.reshape(np.size(imagePoints), 1)

        fMarks = self.camera.fiducials.reshape(np.size(self.camera.fiducials), 1)

        n = int(len(imagePoints))  # number of observations
        u = 6  # 6 orientation parameters

        A = np.zeros((n, u))  # A matrix (n,u)

        j = 0
        for i in range(len(imagePoints)):
            if i % 2 == 0:
                A[i, 0] = 1
                A[i, 1] = 0
                A[i, 2] = fMarks[j]
                A[i, 3] = fMarks[j + 1]
                A[i, 4] = 0
                A[i, 5] = 0
            else:
                A[i, 0] = 0
                A[i, 1] = 1
                A[i, 2] = 0
                A[i, 3] = 0
                A[i, 4] = fMarks[j]
                A[i, 5] = fMarks[j + 1]
                j += 2

        X = np.dot(la.inv(np.dot(np.transpose(A), A)), np.dot(np.transpose(A), imagePoints))
        v = np.dot(A, X) - imagePoints

        adjustment_results = {"params": X, "residuals": v, "N": np.dot(np.transpose(A), A)}

        self.__innerOrientationParameters = X  # updating the inner orientation params

    def ComputeInverseInnerOrientation(self):
        """
        Computes the parameters of the inverse inner orientation transformation
        :return: parameters of the inverse transformation
        :rtype: dict
        """
        a0 = self.innerOrientationParameters[0]
        b0 = self.innerOrientationParameters[1]
        a1 = self.innerOrientationParameters[2]
        a2 = self.innerOrientationParameters[3]
        b1 = self.innerOrientationParameters[4]
        b2 = self.innerOrientationParameters[5]

        mat = np.array([[a1[0], a2[0]], [b1[0], b2[0]]])
        mat = la.inv(mat)

        return np.array([a0[0], b0[0], mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1]]).T

    def ImageToCamera(self, imagePoints):
        """
        Transforms image points to camera points
        :param imagePoints: image points
        :type imagePoints: np.array nx2
        :return: corresponding camera points
        :rtype: np.array nx2
        """
        imagePoints = np.array(imagePoints).T

        inverse_pars = self.ComputeInverseInnerOrientation()

        if imagePoints.size == 2:
            imagePoints = np.reshape(np.array(imagePoints), (np.size(imagePoints), 1))

        T = np.array([[inverse_pars[0]], [inverse_pars[1]]])
        R = np.array([[inverse_pars[2], inverse_pars[3]], [inverse_pars[4], inverse_pars[5]]])

        return (np.dot(R, imagePoints - T)).T


if __name__ == '__main__':
    pass
