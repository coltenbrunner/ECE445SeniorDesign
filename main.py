import socket
import struct
import numpy as np
import threading
import queue
import cv2
import os
import time
import base64
import matplotlib.pyplot as plt
import json


from predict import model
from predict import predict_image


# Raspberry Pi IP address and port for socket communication
RASPBERRY_PI_IP = '172.16.167.160'  # Replace with your Raspberry Pi's IP address
PORT = 3000  # Same port as used in Raspberry Pi



def send_classification_result(classification_result, accuracy, elapsed_time, image):
    try:
        # Convert PyTorch tensor to NumPy array
        image_np = image.cpu().numpy()
        # Convert the array to the correct data type and shape
        image_np = (image_np * 255).astype(np.uint8)
        image_np = np.transpose(image_np, (1, 2, 0))  # Move channel dimension to the last axis
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            while True:
                try:
                    client_socket.connect((RASPBERRY_PI_IP, PORT))
                    break  # If connection successful, break out of the loop
                except Exception as e:
                    print("Error connecting to Raspberry Pi:", e)
                    print("Retrying in 5 seconds...")
                    time.sleep(5)  # Wait for 5 seconds before retrying
                    
            # Convert the image to base64 string
            _, img_encoded = cv2.imencode('.jpg', image_np)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')

            # Construct the classification result dictionary
            classification_data = {
                'classification_result': classification_result,
                'accuracy': accuracy,
                'elapsed_time': elapsed_time,
                'image_base64': img_base64
            }
            
            # Serialize the dictionary to JSON
            classification_data_json = json.dumps(classification_data)

            # Send the JSON data
            client_socket.sendall(struct.pack('<L', len(classification_data_json)))
            client_socket.sendall(classification_data_json.encode('utf-8'))
            
    except Exception as e:
        print("Error connecting to Raspberry Pi:", e)




def save_frame(frame, save_directory, frame_count, frame_saved_event):
    filename = f"frame_{frame_count}.jpg"
    filepath = os.path.join(save_directory, filename)
    try:
        cv2.imwrite(filepath, frame)
        print(f"Frame saved as {filepath}")
        # Signal that the frame has been saved
        frame_saved_event.set()
    except Exception as e:
        print(f"Error saving frame: {e}")

    return filepath

# Function to receive frames from the Raspberry Pi
def receive_frames(frame_queue, save_directory, frame_saved_event):
    frame_data = b''
    frame_count = 0
    print("Connecting to Raspberry Pi...")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((RASPBERRY_PI_IP, PORT))
            print("Connected to Raspberry Pi")
            while True:
                try:
                    # Receive the size of the image frame
                    data_size = client_socket.recv(4)
                    if not data_size:
                        break
                    frame_size = struct.unpack('<L', data_size)[0]
                    
                    # Receive the frame data
                    frame_data = b''
                    while len(frame_data) < frame_size:
                        packet = client_socket.recv(frame_size - len(frame_data))
                        if not packet:
                            break
                        frame_data += packet
                    
                    # Convert the frame data into a numpy array
                    frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                    
                    # Decode the frame
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    
                    # Save the frame as a JPEG image
                    frame_saved_event.clear()  # Clear event before saving frame
                    filepath = save_frame(frame, save_directory, frame_count, frame_saved_event)
                    
                    # Wait for the frame to be saved before putting it into the queue
                    frame_saved_event.wait()
                    
                    # Put the filepath into the queue
                    frame_queue.put(filepath)
                    
                    frame_count += 1
                    
                    print("Frame received")
                except Exception as e:
                    print("Error receiving frame:", e)
    except Exception as e:
        print("Error connecting to Raspberry Pi:", e)

def classify_frames(frame_queue, save_directory):
    counter = 0
    while True:
        try:
            filepath = frame_queue.get(timeout=1)
            print("Received file path:", filepath)  # Verify the file path

            if os.path.exists(filepath):
                # Classify the frame using the predict_image function
                classification_result, accuracy, elapsed_time, im = predict_image(filepath, model, counter, save_directory)
                # Send the classification result back to the Raspberry Pi
                send_classification_result(classification_result, accuracy, elapsed_time, im)
                print("Classifying and predicting attempt")

                # plt.figure(figsize=(10,6))
                # plt.imshow(im)
                # plt.title(f'Prediction: {classification_result}\n Prediction Accuracy: {accuracy}\nElapsed Time: {elapsed_time:.2f} seconds')  # Add title with prediction and elapsed time
                # plt.savefig(os.path.join(save_directory, f'plot_{counter}.png'))  # Save plot with the prediction and elapsed time
                # plt.close()  # Close the plot to avoid displaying it

                counter += 1

            else:
                print(f"Error: File '{filepath}' does not exist")
        except queue.Empty:
            print("No frame in queue yet")
        except Exception as e:
            print("Error classifying frame:", e)

# Main function
if __name__ == "__main__":
    try:
        # Create a queue for passing file paths of frames from the receiving thread to the classification thread
        frame_queue = queue.Queue()

        # Create an event to signal when a frame is saved
        frame_saved_event = threading.Event()

        # Directory to save received frames
        save_directory = "/Users/coltenbrunner/Desktop/ECE445_BirdCAM/bird_classification_script/saved_frames"
        classify_directory = "/Users/coltenbrunner/Desktop/ECE445_BirdCAM/bird_classification_script/classifications"
        # Start receiving frames from Raspberry Pi in a separate thread
        receive_thread = threading.Thread(target=receive_frames, args=(frame_queue, save_directory, frame_saved_event))
        receive_thread.start()

        # Start classifying frames and sending classification result back to Raspberry Pi
        classify_thread = threading.Thread(target=classify_frames, args=(frame_queue, classify_directory))
        classify_thread.start()

        

        # Wait for threads to finish
        receive_thread.join()
        classify_thread.join()

    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    finally:
        cv2.destroyAllWindows()



# Earlier tests had to move this to the main function as it had some unexpected segfault in sendtopi.py
"""from predict import predict_image
from predict import model



saveDirectory = "/Users/coltenbrunner/Desktop/ECE445_BirdCAM/plots"

predict_image('/Users/coltenbrunner/Desktop/ECE445_BirdCAM/Bird-Classification/test-data/bird.jpg', model, 0, saveDirectory)


predict_image('/Users/coltenbrunner/Desktop/ECE445_BirdCAM/Bird-Classification/test-data/bird3.jpg', model, 1, saveDirectory)

#predict_image('/Users/coltenbrunner/Desktop/ECE445_BirdCAM/Bird-Classification/test-data/bird4.jpg', model)

predict_image('/Users/coltenbrunner/Desktop/ECE445_BirdCAM/Bird-Classification/test-data/bird5.jpg', model, 2, saveDirectory)

predict_image('/Users/coltenbrunner/Desktop/ECE445_BirdCAM/Bird-Classification/test-data/Bird6.jpg', model, 3, saveDirectory)

predict_image('/Users/coltenbrunner/Desktop/ECE445_BirdCAM/Bird-Classification/test-data/bird7.jpeg', model, 4, saveDirectory)

predict_image('/Users/coltenbrunner/Desktop/ECE445_BirdCAM/Bird-Classification/test-data/bird8.jpeg', model, 5, saveDirectory)

"""