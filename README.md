# Industry-Proj
016835

### This is a repository for one of my academic undergrad projects 
### This project was a cooperation between myself and the Survey of Israel to create a tool for Camera Calibration through Bundle Adjustment

#### results and output example of the tool:

+------+--------------+---------------+-------------------+
| iter |   v.T * v    | ||dx_camera|| | ||dx_tie_points|| |
+------+--------------+---------------+-------------------+
|  1   | 150332.38700 |  3.89146E+01  |    3.12528E+02    |
|  2   |  1668.08386  |  4.28460E-01  |    1.08870E+02    |
|  3   |   0.01311    |  6.06867E-02  |    3.20509E-01    |
|  4   |   0.00116    |  1.25488E-03  |    1.46612E-04    |
|  5   |   0.00116    |  5.22845E-06  |    9.21897E-07    |
+------+--------------+---------------+-------------------+
+--------------------------------+--------------+--------------+
|         IO Parameters          |    Value     |     STD      |
+--------------------------------+--------------+--------------+
|          Focal Length          | 111.4699(mm) | ±0.2729(mm)  |
|     PRINCIPAL POINT OFFSET     |      -       |      -       |
|               XP               |  0.0207(mm)  | ±0.0186(mm)  |
|               YP               | -0.0042(mm)  | ±0.0280(mm)  |
|       RADIAL DISTORTIONS       |      -       |      -       |
|               K1               | 3.77236E-08  | ±2.42843E-08 |
|               K2               | -4.37100E-12 | ±1.40776E-11 |
|               K3               | -3.66789E-16 | ±2.47528E-15 |
|     TANGENTIAL DISTORTIONS     |      -       |      -       |
|               P1               | -1.65763E-07 | ±7.40781E-08 |
|               P2               | -2.34564E-07 | ±7.77047E-08 |
| AFFINITY AND NON-ORTHOGONALITY |      -       |      -       |
|               B1               | -7.16204E-06 | ±4.42602E-06 |
|               B2               | 1.91182E-05  | ±3.39184E-06 |
+--------------------------------+--------------+--------------+
Execution time in minutes: 0.09894011020660401
