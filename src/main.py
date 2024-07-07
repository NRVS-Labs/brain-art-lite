import streamlit as st
import brainflow
import os
from assets import board_id_pairs, read_userdata, update_userdata
from brainflow.data_filter import DataFilter
import numpy as np

# local imports
# from communications import Comms

import brainflow
from brainflow import BoardIds, BrainFlowInputParams, BoardShim, BrainFlowError, BrainFlowClassifiers, BrainFlowMetrics, DataFilter

from plotting import generate_raw_plot

import time

import pandas as pd

# Prototype methods for a web-browser interface for Brain Generated Artwork
# - Leonardo Ferrisi

class BrowserUI:

    def __init__(self, title="Brain Generated Artwork Prototype", debug=False):
        self.debug = debug
        self.userdata = read_userdata()

        self.current_working_directory = os.getcwd()
        self.current_file_path = os.path.abspath(__file__)
        self.current_file_path_directory = os.path.dirname(self.current_file_path)


        from PIL import Image
        logo_path = os.path.join(self.current_file_path_directory, "local_assets", "logo.png")

        LOGO = Image.open(logo_path)
        col1, col2, col3 = st.columns([5, 5, 5])
        col2.image(image=LOGO, caption="NRVS Labs", width=200)
        # st.image(image=LOGO, caption="NRVS Labs", width=200)
        st.title(title)
        st.info('This prototype exists for demonstrating art generation methods from EEG data. Pre-recorded data in the proper csv format is required.')

        st.info('The current art generation method of this web app is a CPPN art generator')
        
        if 'connected' not in st.session_state:
            st.session_state['connected'] = False

        self.connected = st.session_state['connected']
        
        if 'streaming' not in st.session_state:
            st.session_state['streaming'] = False
        self.streaming = st.session_state['streaming']

        if 'recording_data' not in st.session_state:
            st.session_state['recording_data'] = None

        if 'EEG_DATA' not in st.session_state:
            st.session_state['EEG_DATA'] = None
        self.EEG_DATA = st.session_state['EEG_DATA']

        if 'image_name' not in st.session_state:
            st.session_state['image_name'] = None
        self.image_name = st.session_state['image_name']

        if 'EEG_CHANNELS' not in st.session_state:
            st.session_state['EEG_CHANNELS'] = None
        self.EEG_CHANNELS = st.session_state['EEG_CHANNELS']

        if "collect_data" not in st.session_state:
            st.session_state["collect_data"] = False
        self.collect_data = st.session_state["collect_data"]

        if 'USER_DATA' not in st.session_state:
            st.session_state['USER_DATA'] = None
        self.USER_DATA = st.session_state['USER_DATA']

        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()

        self.show_prompts()

    
    def connect_board(self, board_id, port=None):
        """
        Connect a board and initiate it as a BoardShim Object
        """
        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()

        if port != None: params.serial_port = port
        
        if ('BOARD' not in st.session_state) or (st.session_state['BOARD'] == None):
            st.session_state['BOARD'] = BoardShim(int(board_id), params)

        self.BOARD = st.session_state['BOARD']
        self.BOARD.prepare_session()
        
        st.session_state['EEG_CHANNELS'] = self.BOARD.get_eeg_channels(int(board_id))
        self.EEG_CHANNELS = st.session_state['EEG_CHANNELS']

        try:
            with st.expander("[For Debugging] Session State:"):
                st.write(st.session_state)
            with st.expander("[For Debugging] Board Description:"):
                board_descr = BoardShim.get_board_descr(board_id)
                st.write("Board Description: ", board_descr)
        except:
            st.text("No session state available. or failed to print :(")
    def disconnect_board(self):
        """
        Disconnect the board and set the user data to None
        """
        if self.connected:
            self.BOARD = st.session_state['BOARD']
            self.BOARD.release_all_sessions()
            st.session_state['BOARD'] = None
            self.connected = False

    def change_connection_status(self):
        """
        Change the connection status of the board
        """
        if self.connected:
            self.connected = False
        else: 
            self.connecred = True

    def stream_data(self):
        """
        Stream data from an initialized BoardShim Object
        """
        self.BOARD = st.session_state['BOARD']
        self.BOARD.start_stream()

        # Add a progress bar
        progress_bar = st.progress(0)
        durations = self.collection_time
        iterations = 100
        sleep_time = durations / iterations
        start_time = time.time()
        for i in range(iterations):
            time_elapsed = time.time() - start_time
            label = f"{time_elapsed:.2f} seconds elapsed"
            progress_bar.progress(i + 1, text=label)
            time.sleep(sleep_time)

        user_data = self.BOARD.get_board_data()

        # Overwrite EEG_DATA
        st.session_state['EEG_DATA'] = user_data

        self.BOARD.stop_stream()

        st.success("Task completed! Data collected.")
        st.info("If you would like to record again, please click the **R**-key to re-run before clicking `START` again.")
        return user_data
    
    def generate_feature_vector(self, data):
        """
        Generate a vector of features of data
        """
        from preprocessing import get_simple_feature_vector
        
        feature_vector = get_simple_feature_vector(data=data, boardID=self.CURRENT_BOARD_ID)

        return feature_vector

    def generate_image(self, features, resolution=(512, 512), num_layers=10, layer_width=18, activation_name="tanh", input_scalar_1=1.0, input_scalar_2=1.0, verbose=True):
        """
        Generates an image from features of EEG
        """
        from generation import NumpyArtGenerator

        if self.image_name == "": 
            # Make a generic image name based on the data and time
            # get the date
            date = time.strftime("%Y_%m_%d_%H_%M_%S") 
            generic_image_name = str(date)
            image_name = f"{generic_image_name}.jpg"
        else: 
            image_name = f"{self.image_name}.jpg"
        n = NumpyArtGenerator(resolution=resolution, 
                              feature_vector=features, 
                              num_layers=num_layers, 
                              layer_width=layer_width, 
                              input_scalar_1=input_scalar_1, 
                              input_scalar_2=input_scalar_2, 
                              activation_name=activation_name)
        n.run(verbose=True)
        image_path = n.save_image(image_name, self.image_directory)
        return image_path        
    
    def save_data(self, userdata:dict, data):

        import csv
        import os

        current_filepath = os.path.realpath(__file__)
        current_directory = os.path.dirname(current_filepath)
        output_directory = os.path.join(current_directory, ".." , 'data')
        datetime = time.strftime("%Y_%m_%d_%H_%M_%S")

        def create_folder(folder_path):
            try:
                os.mkdir(folder_path)
                print(f"Folder created successfully: {folder_path}")
            except FileExistsError:
                print(f"Folder already exists: {folder_path}")
            except Exception as e:
                print(f"Error occurred while creating folder: {e}")

        def save_dict_to_csv(dictionary, filename):
            keys = dictionary.keys()
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()
                writer.writerow(dictionary)


        new_output_folder = os.path.join(output_directory, datetime)
        create_folder(new_output_folder)

        output_userdata_path = os.path.join(new_output_folder, f'{datetime}_userdata.csv')
        # Save dictionary as CSV
        save_dict_to_csv(userdata, output_userdata_path)

        # Save brainflow data
        data_file_path = os.path.join(new_output_folder, f'{datetime}_recording_data.csv')
        DataFilter.write_file(data,data_file_path, 'w')
        
        print(f"File Saved at {new_output_folder}")

    def show_prompts(self):
        """
        Show all prompts for starting the Brain Artwork program
        """

        # Show buttons for using either prerecorded data or live data
        # data_source = st.radio("Data Source", ("Live", "Prerecorded"), horizontal=True, help="Select the source of EEG data")

        data_source = "Prerecorded"
        
        # If prerecorded data is selected, show a file upload button
        if data_source == "Prerecorded":
            uploaded_file = st.file_uploader("Choose a file", type="csv")

        if uploaded_file is not None:
            st.write("file uploaded.")
            
        # if data_source == "Live":
            # st.write("Live data selected.")
            
            # Continue with Brain Art - Generation. Backend MUST be running.

            # --------------------------------------------------------------
            # Board selection:

            board_select = st.selectbox(
                'Which board are you using?',
                ('Synthetic', 'Muse2', 'Muse2016', 'OpenBCI Cyton (8-channels)', 'OpenBCI (16-channels)', 'OpenBCI Ganglion', ), 
                help="Select the board you are using to record EEG data.")

            st.write('You selected:', board_select)

            if self.debug:
                st.write("Board ID:", board_id_pairs[board_select]["id"])
            #     st.write("Using port?:", board_id_pairs[board_select]["using_port"])

            self.CURRENT_BOARD = board_select

            # # --------------------------------------------------------------
            # # Port Selection:
            # self.requires_port = board_id_pairs[board_select]["using_port"]
            # if self.requires_port:
            #     st.write("The current board requires a port to be specified.")
            #     current_port = self.userdata["PORT"]

            #     if self.debug: st.write("Port:", current_port)

            #     port_select = st.text_input("The currently saved port is: ", current_port, help="Enter the USB Serial port your board is connected to.")

            #     if port_select != current_port:
            #         try: 
            #             update_userdata("PORT", port_select)
            #         except:
            #             st.write("Error updating port in userdata.dat. Using locally defined PORT")

            #     self.PORT = port_select

            # --------------------------------------------------------------
            # Configure Image Output Directory
            default_image_directory = os.path.join(self.current_working_directory, "local_gen")

            self.image_directory = st.text_input("Current Image Output Directory: ", default_image_directory, help="Enter the directory where you would like to save generated images. Change nothing if you are fine with the default configuration.")

            # --------------------------------------------------------------

        
            # port_param = None
            # if self.requires_port: port_param = self.PORT
            self.CURRENT_BOARD_ID = board_id_pairs[self.CURRENT_BOARD]["id"]
            
            # uploaded_data = pd.read_csv(uploaded_file)

            
            with open("temp.csv", 'wb') as twf:
                twf.write(uploaded_file.getvalue())

            # uploaded_data_raw = DataFilter.read_file(uploaded_file.getvalue())

            uploaded_data_raw = DataFilter.read_file('temp.csv')

            # st.write("TEMP")

            # uploaded_data = DataFilter.read_file('test.csv')
            uploaded_data = pd.DataFrame(np.transpose(uploaded_data_raw))

            self.EEG_CHANNELS = len(uploaded_data.columns)

            uploaded_data = uploaded_data.iloc[:, 1:]

            data = uploaded_data

            #
            #
            
            st.markdown("### Raw Data: ")
            st.info("The following is the raw data output from the board. This is the data that is processed \
                further used to generate the artwork.")

            try:
                with st.expander("Checkout raw data."):
                    st.write("#### Raw Data")
                    # st.info(f"The y axis is the channels in all data. The x axis is the time samples. Note that we are only intersted in the first {len(self.EEG_CHANNELS)} channels as those correspond to EEG. The rest are Accelorometer Data, Battery Level, and more!")
                    
                    st.dataframe(data)

            except Exception as e:
                st.error(e)

            # #
            # #

            # Generate Plot:

            descale_weight = 1000

            # if self.CURRENT_BOARD.lower() == "synthetic":
            #     descale_weight = 1000
            
            fig, ax = generate_raw_plot(boardID=self.CURRENT_BOARD_ID, data=data, transpose=True, title="Raw EEG Data Plot", show=False, save=False, descale_weight=descale_weight, filename="raw_plot.png", show_progress=False)

            # Display the plot in Streamlit
            
            st.pyplot(fig)
            st.info("This is what the data looks like before we filter it and turn it into an image!")
            
            st.divider()

            # Get the features from data

            feature_vector = self.generate_feature_vector(data.transpose())

            print(f"Feature Vector: {feature_vector}")
            # Display the feature vector
            st.write("#### Feature Vector")
            st.info("This is the feature vector that is used to generate the artwork. It is a 1D array of numbers that represent the data collected from the board.")

            gamma_band = feature_vector[4]

            progress_bar = st.progress(0, text="Gamma Band")
            for i in range(int(gamma_band * 100)):
                progress_bar.progress(i, text=f"Gamma Band: {i}%")

            beta_band = feature_vector[1]

            progress_bar = st.progress(0, text="Beta Band")
            for i in range(int(beta_band * 100)):
                progress_bar.progress(i, text=f"Beta Band: {i}%")

            alpha_band = feature_vector[0]

            progress_bar = st.progress(0, text="Alpha Band")
            for i in range(int(alpha_band * 100)):
                progress_bar.progress(i, text=f"Alpha Band: {i}%")

            theta_band = feature_vector[2]
            
            progress_bar = st.progress(0, text="Theta Band")
            for i in range(int(theta_band * 100)):
                progress_bar.progress(i, text=f"Theta Band: {i}%")

            delta_band = feature_vector[3]

            progress_bar = st.progress(0, text="Delta Band")
            for i in range(int(delta_band * 100)):
                progress_bar.progress(i, text=f"Delta Band: {i}%")

            concentration_prediction = feature_vector[5]

            progress_bar = st.progress(0, text="Concentration Prediction: 0%")
            for i in range(int(concentration_prediction * 100)):
                progress_bar.progress(i, text=f"Concentration Prediction: {i}%")

            mindfulness_prediction = feature_vector[6]

            progress_bar = st.progress(0, text="Mindfulness Prediction: 0%")
            for i in range(int(mindfulness_prediction * 100)):
                progress_bar.progress(i, text=f"Mindfulness Prediction: {i}%")

            relaxation_prediction = feature_vector[7]

            progress_bar = st.progress(0, text="Relaxation Prediction: 0%")
            for i in range(int(relaxation_prediction * 100)):
                progress_bar.progress(i, text=f"Relaxation Prediction: {i}%")

            # # Handle saving data
            # # TODO: Handle Saving data
            # if self.collect_data:
            #     self.save_data(self.USER_DATA, data)
            input_scalar_01 = st.slider(label="Placeholder scalar #1", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
            input_scalar_02 = st.slider(label="Placeholder scalar #2", min_value=0.0, max_value=1.0, value=1.0, step=0.01)

            self.image_name = st.text_input("Enter the name of the image you want to generate: ", help="Enter the name of the image you want to generate. This will be used to save the image.")    
            st.session_state['image_name'] = self.image_name
            
            st.divider()
            
            if st.button("Generate image"):
                if self.image_name not in ["None", ""]:
            
                    st.write("Image will be generated below, this may take a moment.")

                    image_path = self.generate_image(resolution=(1920, 1080), 
                                                    features=feature_vector, 
                                                    num_layers=10, 
                                                    layer_width=9, 
                                                    activation_name="tanh", 
                                                    input_scalar_1=input_scalar_01,
                                                    input_scalar_2=input_scalar_02,
                                                    verbose=True)

                    # Handle displaying image
                    self.display_artwork(image_path)
                    
                    # # Handle saving image
                    st.divider()                
                    with open(image_path, "rb") as file:
                        btn = st.download_button(
                                label="Download image",
                                data=file,
                                file_name=f"{self.image_name}.png",
                                mime="image/png"
                            )


    def display_artwork(self, image_path):
        from PIL import Image
        image = Image.open(image_path)
        st.image(image, caption=f"Generated Image: {self.image_name}", width=700)

if __name__ == "__main__":
    webapp = BrowserUI(title="Brain Generated Artwork Prototype")