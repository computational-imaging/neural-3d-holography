"""
Test code for Canon EF lens

Contact: Suyeon Choi (suyeon@stanford.edu)
-----

$ python lens_control.py --index=$i
"""

import serial
import time
import serial, time
import random
import configargparse

# Command line argument processing
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add_argument('--index', type=int, default=None, help='index of plane')

opt = p.parse_args()


t0 = time.perf_counter()
arduino = serial.Serial('COM7', 9600, timeout=1.)
time.sleep(0.1)  # give the connection a second to settle
print(f' -- connection is established.. {time.perf_counter() - t0}')


if opt.index is not None:
    time.sleep(5)
    print(f' -- writing...{opt.index}')
    arduino.write(f'{opt.index}'.encode())

    time.sleep(5)

    data = arduino.readline().decode('UTF-8')  # hear back from your arduino
    if data:
        print(data)
else:
    while True:
        my_input = input()
        arduino.write(f'{str(my_input)}'.encode())
        time.sleep(1.0)

        data = arduino.readline().decode('UTF-8')
        if data:
            print(data)
