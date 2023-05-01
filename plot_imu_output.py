import matplotlib.pyplot as plt
import numpy as np
import serial as s
from scipy.signal import lfilter
import json as js

ser = s.Serial("/dev/ttyACM0", timeout=5)
print(ser.name)

acc_x = []
acc_y = []
acc_z = []
g_x = []
g_y = []
g_z = []
while True:
    line = ser.readline().decode('utf-8').strip()
    if len(line) > 0 and line[0] == "{":
        data = js.loads(line)
        print(data)
        acc_x.append(data["acc_x"])
        acc_y.append(data["acc_y"])
        acc_z.append(data["acc_z"])
        g_x.append(data["gyro_x"])
        g_y.append(data["gyro_y"])
        g_z.append(data["gyro_z"])
    if len(acc_x) > 5000:
        break

n = np.arange(0, len(acc_x))
print("AccX Avg: ", np.average(acc_x))
print("AccY Avg: ", np.average(acc_y))
print("AccZ Avg: ", np.average(acc_z))
print("GyroX Avg: ", np.average(g_x))
print("GyroY Avg: ", np.average(g_y))
print("GyroZ Avg: ", np.average(g_z))
h = [
    0.000000000000000000,
    -0.000007087403427004,
    0.000029767502584404,
    0.000114814214174261,
    0.000000000000000000,
    -0.000366353814124796,
    -0.000352683625541193,
    0.000521603757357347,
    0.001201752486965034,
    -0.000000000000000002,
    -0.002244988547909949,
    -0.001838240359041174,
    0.002395800615580116,
    0.004981724885096919,
    -0.000000000000000005,
    -0.007947470657534797,
    -0.006117762657107704,
    0.007565194395564235,
    0.015053603853409040,
    -0.000000000000000010,
    -0.022569549930043613,
    -0.017090145909041038,
    0.021043826743725257,
    0.042354268914770697,
    -0.000000000000000014,
    -0.070049904408009608,
    -0.059715968340122418,
    0.091763199562286291,
    0.301275623529488390,
    0.399997950381802969,
    0.301275623529488445,
    0.091763199562286291,
    -0.059715968340122418,
    -0.070049904408009595,
    -0.000000000000000014,
    0.042354268914770718,
    0.021043826743725264,
    -0.017090145909041034,
    -0.022569549930043616,
    -0.000000000000000010,
    0.015053603853409045,
    0.007565194395564238,
    -0.006117762657107704,
    -0.007947470657534804,
    -0.000000000000000005,
    0.004981724885096921,
    0.002395800615580119,
    -0.001838240359041176,
    -0.002244988547909948,
    -0.000000000000000002,
    0.001201752486965033,
    0.000521603757357347,
    -0.000352683625541193,
    -0.000366353814124796,
    0.000000000000000000,
    0.000114814214174261,
    0.000029767502584404,
    -0.000007087403427004,
    0.000000000000000000,
]
h_c = 29
acc_x_lp = lfilter(h, 1.0, acc_x)
acc_y_lp = lfilter(h, 1.0, acc_y)
acc_z_lp = lfilter(h, 1.0, acc_z)
g_x_lp = lfilter(h, 1.0, g_x)
g_y_lp = lfilter(h, 1.0, g_y)
g_z_lp = lfilter(h, 1.0, g_z)

plt.subplot(221)
plt.plot(n, acc_x, label = "Acc X")
plt.plot(n, acc_y, label = "Acc Y")
plt.plot(n, g_x, label = "Gyro X")
plt.plot(n, g_y, label = "Gyro Y")
plt.plot(n, g_z, label = "Gyro Z")
plt.legend()
plt.subplot(222)
plt.plot(n, acc_z, label = "Acc Z")
plt.legend()
plt.subplot(223)
plt.plot(n, acc_x_lp, label = "Acc X LP")
plt.plot(n, acc_y_lp, label = "Acc Y LP")
plt.plot(n, g_x_lp, label = "Gyro X LP")
plt.plot(n, g_y_lp, label = "Gyro Y LP")
plt.plot(n, g_z_lp, label = "Gyro Z LP")
plt.legend()
plt.subplot(224)
plt.plot(n, acc_z_lp, label = "Acc Z LP")
plt.legend()
plt.show()
