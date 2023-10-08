import serial
import time
ser = serial.Serial('/dev/ttyUSB0', 9600)

for i in range(50):
    ser.write(b'\xA0\x01\x00\xA1')
    time.sleep(0.02)
    ser.write(b'\xA0\x01\x01\xA2')
    time.sleep(0.02)

ser.close()