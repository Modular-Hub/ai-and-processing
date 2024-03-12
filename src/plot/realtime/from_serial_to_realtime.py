import serial
import matplotlib.pyplot as plt
from drawnow import drawnow

# Specify the serial port (COM5) and baud rate
serial_port = serial.Serial('COM5', baudrate=115200, timeout=1)

data_buffer = []

# Function for updating the plot in real-time
def plot_data():
    print("plot-data")
    plt.plot(data_buffer, label='Real-time Data')
    plt.xlabel('Sample Number')
    plt.ylabel('Value')
    plt.title('Real-time Data Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

plt.ion()

try:
    while True:
        data = serial_port.read().decode('utf')

        if len(data) > 0:
            print(data, type(data))
            data = data[:-1]
            data = [i for i in map(int, data.split(','))]

            print(type(data), data)
            # try:
            #     value = int(data)

            #     # Append the value to the data buffer
            #     # data_buffer.append(value)

            #     # # Update the plot in real-time
            #     # drawnow(plot_data)

            # except ValueError:
            #     print(f"Invalid data: {data}")



except KeyboardInterrupt:
    print("Program terminated by user.")

finally:
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Display the final plot
    # Close the serial port when done
    serial_port.close()
