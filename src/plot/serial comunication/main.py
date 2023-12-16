import serial
import os

file_name = 'test.csv'

serial_inst = serial.Serial()
serial_inst.port = '/dev/ttyACM0'
serial_inst.baudrate = 115200

serial_inst.open()
while True:
    if serial_inst.in_waiting:
        value = serial_inst.read()
        print(value.decode('utf'))

        file = open(file_name,'ab')
        file.write(value)
