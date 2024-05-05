import io
import logging
import socketserver
from http import server
from threading import Condition, Thread, Event
import sys
import time
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
import socket
import signal
import gpiod
import threading
import csv
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend (non-GUI)
import matplotlib.pyplot as plt
import shutil
import os
import datetime
import struct
import numpy as np
import base64
import cv2
import json


logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('picamera2').setLevel(logging.WARNING)

# Paths
#csv_file_path = 'recordings.csv'
csv_file_path = 'demo.csv'
html_file_path = 'index.html'
#graph_image_path = 'recordings.png'
graph_image_path = 'demo.png'

# Global variables
streaming_enabled = False
is_recording = False
latest_classification_result = ""

# Raspberry Pi IP address and port for socket communication
HOST = '172.16.167.160'  # Raspberry Pi IP address
PORT = 3000  # Choose a port

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (1536, 864)}))
encoder1 = MJPEGEncoder(10000000)
#encoder2 = H264Encoder(10000000)



# Function to copy data from output to output_buffer

def capture_frames():
    global is_recording
    try:
        while True:
            if is_recording:  # Check if recording is enabled
                selected_frame = output.frame  # Get the latest frame from output
                if selected_frame is not None:
                    frame_array = np.frombuffer(selected_frame, dtype=np.uint8)
                    # Check if the frame array is not empty or invalid
                    if frame_array.size > 0:
                        try:
                            size = len(selected_frame)
                            conn.sendall(struct.pack('<L', size))  # Send the size of the frame
                            conn.sendall(selected_frame)  # Send the frame over the socket connection
                            print("Selected frame:", frame_array)  # Print the selected frame for debugging
                            time.sleep(20)  # Delay between sending frames
                        except BrokenPipeError:
                            #print("Failed to send frame. Discarding...")
                            pass  # Skip sending this frame
                            time.sleep(20)
                    else:
                        print("Invalid frame detected. Discarding...")
                else:
                    print("No frame available. Skipping...")
            else:
                time.sleep(5)  # Sleep if not recording to reduce CPU usage
    except Exception as e:
        logging.error("Error capturing frame: %s", e)






# Function to start capturing frames from the camera and sending them to the laptop
def start_sending_frames():
    try:
        capture_thread = threading.Thread(target=capture_frames, args=())
        capture_thread.daemon = True
        capture_thread.start()
    except Exception as e:
        print("Error starting capture thread:", e)

latest_classification_result = {
    'prediction_string': 'N/A',
    'accuracy': 'N/A',
    'elapsed_time': 'N/A'
}

# Function to receive classification results and update latest_classification_result
def receive_classification_results(conn):
    try:
        while True:
            # Receive data size
            data_size_bytes = conn.recv(4)
            if not data_size_bytes:
                break
            data_size = struct.unpack('<L', data_size_bytes)[0]

            # Receive data
            data = b''
            while len(data) < data_size:
                packet = conn.recv(data_size - len(data))
                if not packet:
                    break
                data += packet

            # Decode JSON data
            classification_result = json.loads(data.decode('utf-8'))

            # Extract classification information
            prediction_string = classification_result.get('classification_result', 'N/A')
            accuracy = classification_result.get('accuracy', 'N/A')
            elapsed_time = classification_result.get('elapsed_time', 'N/A')

            # Update latest classification result
            latest_classification_result['prediction_string'] = prediction_string
            latest_classification_result['accuracy'] = accuracy
            latest_classification_result['elapsed_time'] = elapsed_time

            # Display classification information
            print("Prediction String:", prediction_string)
            print("Accuracy:", accuracy)
            print("Elapsed Time:", elapsed_time)
            
            update_web_server()
    except ConnectionResetError:
        print("Connection reset by peer")
    except json.JSONDecodeError:
        print("Error decoding JSON data")
    except Exception as e:
        print("Error receiving classification results:", e)





# Function to update the server with the new HTML content
def update_server_with_html(html_content):
    try:
        # Write the HTML content to the index.html file
        with open(html_file_path, 'w') as html_file:
            html_file.write(html_content)

        print("HTML content updated successfully")
    except Exception as e:
        print("Error updating HTML content:", e)

# Update the web server with the new HTML content after receiving a classification result
def update_web_server():
    # Generate HTML page with the latest classification result
    html_content = generate_html_page(latest_classification_result)
    # Update the web server with the new HTML content
    update_server_with_html(html_content)

# Function to run update_web_server() after a delay
def delayed_update(delay):
    time.sleep(delay)
    generate_graph()
    print("Generated new Graph")


# Function to save timestamp to CSV file
def save_timestamp_to_csv():
    if is_recording:
        timestamp = datetime.datetime.now()
        with open(csv_file_path, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp])
        time.sleep(5)
def generate_graph():
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path, header=None, names=['Timestamp'])

        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("CSV file is empty")

        # Convert 'Timestamp' column to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # # Filter out timestamps older than 10 minutes
        # ten_minutes_ago = datetime.datetime.now() - datetime.timedelta(minutes=10)
        # df = df[df['Timestamp'] >= ten_minutes_ago]

        # Check if there are still timestamps within the last 10 minutes
        if df.empty:
            print("No recordings within the last 10 minutes.")
            return

        # Aggregate data by minute
        df['Minute'] = df['Timestamp'].dt.floor('1Min')
        grouped = df.groupby('Minute').size()

        # Plot the bar graph
        grouped.plot(kind='bar')
        plt.xlabel('Time')
        plt.ylabel('Recordings')
        plt.title('Recordings per Time Interval')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the graph as an image
        plt.savefig(graph_image_path)
        print("Graph generated successfully.")
    except Exception as e:
        print("Error:", e)

# def generate_graph():
#     try:
#         # Read the CSV file
#         df = pd.read_csv(csv_file_path, header=None, names=['Timestamp'])

#         # Check if DataFrame is empty
#         if df.empty:
#             raise ValueError("CSV file is empty")

#         # Convert 'Timestamp' column to datetime
#         df['Timestamp'] = pd.to_datetime(df['Timestamp'])

#         # Filter out timestamps older than 10 minutes
#         ten_minutes_ago = datetime.datetime.now() - datetime.timedelta(minutes=10)
#         df = df[df['Timestamp'] >= ten_minutes_ago]

#         # Check if there are still timestamps within the last 10 minutes
#         if df.empty:
#             print("No recordings within the last 10 minutes.")
#             return

#         # Aggregate data by minute
#         df['Minute'] = df['Timestamp'].dt.floor('1Min')
#         grouped = df.groupby('Minute').size()

#         # Plot the bar graph
#         grouped.plot(kind='bar')
#         plt.xlabel('Time')
#         plt.ylabel('Recordings')
#         plt.title('Recordings per Time Interval (Last 10 Minutes)')
#         plt.xticks(rotation=45)
#         plt.tight_layout()

#         # Save the graph as an image
#         plt.savefig(graph_image_path)
#         print("Graph generated successfully.")
#     except Exception as e:
#         print("Error:", e)


""" This was to generate the graph for day which is still useful but not right now
# Function to generate bar graph from CSV data
def generate_graph():
    df = pd.read_csv(csv_file_path, header=None, names=['Timestamp'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date
    grouped = df.groupby('Date').size()

    # Plot the bar graph
    grouped.plot(kind='bar')
    plt.xlabel('Date')
    plt.ylabel('Recordings')
    plt.title('Recordings per Time Interval')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the graph as an image
    plt.savefig(graph_image_path)
"""


# Function to toggle streaming
def toggle_streaming():
    global streaming_enabled, is_recording
    streaming_enabled = not streaming_enabled
    if streaming_enabled:
        picam2.start_recording(encoder1, FileOutput(output))
        #picam2.start_recording(encoder2, 'test.mp4')
        is_recording = True
    else:
        picam2.stop_recording()
        is_recording = False

# Function to listen for user input and toggle streaming
def input_listener():
    print("Press Enter to toggle streaming...")
    while True:
        input()  # Wait for user to press Enter
        toggle_streaming()
        print("Streaming enabled:", streaming_enabled)

# Function to listen for signals from GPIO button
def signal_listener():
    global streaming_enabled, is_recording
    BUTTON_PIN = 24
    chip = gpiod.Chip('gpiochip4')
    button_line = chip.get_line(BUTTON_PIN)
    button_line.request(consumer="Button", type=gpiod.LINE_REQ_DIR_IN)
    while True:
        button_state = button_line.get_value()
        print("Button state:", button_state)
        if button_state and not is_recording:
            toggle_streaming()
            print("Streaming enabled:", streaming_enabled)
            print("Is recording?:", is_recording)
            #event.set()
            time.sleep(2)
        elif is_recording and not button_state:
            toggle_streaming()
            print("Streaming enabled:", streaming_enabled)
            print("Is recording?:", is_recording)
            #event.set()
            time.sleep(2)
        else:
            print("Streaming enabled:", streaming_enabled)
            print("Is recording?:", is_recording)
            time.sleep(5)

def generate_html_page(latest_classification_result):
    html_content = """\
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <meta http-equiv="refresh" content="5"> <!-- Updated to refresh every 5 seconds -->
            <title>Raspberry Pi Web Dashboard</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f4f4f4;
                }
                .container {
                    max-width: 800px;
                    margin: 20px auto;
                    padding: 20px;
                    background-color: #fff;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    text-align: center;
                }
                h1, h2 {
                    color: #333;
                }
                img {
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin-top: 10px;
                    margin-left: auto;
                    margin-right: auto;
                }
                .classification-info {
                    margin-top: 20px;
                    text-align: left;
                }
                .graph {
                    margin-top: 20px;
                    text-align: center;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Raspberry Pi Web Dashboard</h1>
    """

    if streaming_enabled:
        html_content += """
                <div class="live-stream">
                    <h2>Live Video Stream</h2>
                    <img src="/stream.mjpg" alt="Live Stream">
                </div>
        """

    html_content += """
                <div class="classification-info">
                    <h2>Classification Result</h2>
                    <p><strong>Prediction String:</strong> <span id="prediction-string">{prediction_string}</span></p>
                    <p><strong>Accuracy:</strong> <span id="accuracy">{accuracy}</span></p>
                    <p><strong>Elapsed Time:</strong> <span id="elapsed-time">{elapsed_time}</span> seconds</p>
                </div>
                <div class="graph">
                    <h2>Motion over Time</h2>
                    <img src="data:image/png;base64,{base64_data}" alt="Motion Graph">
                </div>
            </div>
        </body>
        </html>
    """.format(prediction_string=latest_classification_result.get('prediction_string', 'N/A'),
               accuracy=latest_classification_result.get('accuracy', 'N/A'),
               elapsed_time=latest_classification_result.get('elapsed_time', 'N/A'),
               base64_data=get_base64_encoded_image())

    return html_content



def get_base64_encoded_image():
    with open("/home/colten/ECE445_BirdCAM/demo.png", "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


# Function to fetch the latest classification result
def get_latest_classification_result():
    global latest_classification_result
    return latest_classification_result

# StreamingOutput class for buffering camera stream
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)  # Initialize condition with the lock

    def write(self, buf):
        with self.lock:
            self.frame = buf
            self.condition.notify_all()




# StreamingHandler class for handling HTTP requests
class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            # Pass the latest classification result to generate_html_page
            content = generate_html_page(latest_classification_result).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            if streaming_enabled:
                self.send_response(200)
                self.send_header('Age', 0)
                self.send_header('Cache-Control', 'no-cache, private')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
                self.end_headers()
                try:
                    while True:
                        with output.condition:
                            output.condition.wait()
                            frame = output.frame
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                except ConnectionError:
                    pass
                    #logging.warning('Client disconnected unexpectedly')
                except Exception as e:
                    logging.warning('Error handling client request: %s', str(e))
            else:
                self.send_response(204)  # No content response
                self.end_headers()
        else:
            self.send_error(404)
            self.end_headers()

# StreamingServer class for serving the stream
class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True




if __name__ == "__main__":
    try:
        # Initialize streaming server
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        server_thread = Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        



        output = StreamingOutput()
        # Start the input listener thread
        # input_thread = Thread(target=input_listener)
        # input_thread.daemon = True
        # input_thread.start()

        # Start the signal listener thread
        signal_thread = Thread(target=signal_listener, args=())
        signal_thread.daemon = True
        signal_thread.start()

        # Start a thread to run the update after a delay
        #delayed_update_thread = threading.Thread(target=delayed_update, args=(40,))
        #delayed_update_thread.start()
        
        # Start a thread to run the update after a delay
        delayed_update_thread = threading.Thread(target=delayed_update, args=(20,))
        delayed_update_thread.daemon = True
        delayed_update_thread.start()

        # Start receiving classification results
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((HOST, PORT))
            server_socket.listen()

            print("Waiting for connection...")
            try:
                while True:
                    conn, addr = server_socket.accept()
                    print("Connected to:", addr)

                    classification_thread = threading.Thread(target=receive_classification_results, args=(conn,))
                    classification_thread.daemon = True
                    classification_thread.start()

                    # capture_thread = threading.Thread(target=capture_frames, args=())
                    # capture_thread.daemon = True
                    # capture_thread.start()

                    # CSV storage
                    #save_timestamp_to_csv()


                    start_sending_frames()
                    #event.wait()  # Wait for the signal to start/stop streaming
                    #event.clear()  # Reset the event
            finally:
                server_socket.close()
      
    except KeyboardInterrupt:
        print("Interrupt from keyboard")
    finally:
        if picam2:
            picam2.stop_recording()

# # Main function
# if __name__ == "__main__":
#     initialize_server()
#                <meta http-equiv="refresh" content="5">