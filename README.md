README
    This is the README for the 'inference.py' script used for processing data and running the simple transformer model for elephant call prediction. 

    This script runs the inference prediction pipeline.
    The prediction pipeline consists of 2 main components:

        1) Predict call for each second using the transformer model.

        2) Elephant call prediction. Output the models
        predictions for start / end times of calls 
        in txt format.

    The script allows for complete and partial runs of the 2 steps above:

        - Flag '--make_full_preds': Process the '.wav' files (step 1).

        - Flag '--save_calls': Generate elephant call predictions (step 2). 

    Additional command line args:

        - Flag '--model': Path to transformer model

        - Flag '--data_path': Path to directory where audio files are stored.

        - Flag '--preds_path': Path ot directory where prediction arrays are stored.

        - Flag '--call_preds_path': Path to directory where selection tables are stored.

        - Flag '--device': device to run model on i.e. 'cpu' or 'gpu'. Note: if gpu is not cuda supported, it might throw an error.
    
    For each part of the pipeline, command line arguments are used to properly specify necessary data, model path and device to run on (cpu or gpu).

KEY FILES
    This folder contains all of the python scripts needed to run the primary script 'inference.py'. Additionally, the folder 'Model/' contains the model. 

    When you run the script, unless otherwise specified, the default location for output values are as follows: (this steps need to be confirmed, so we recommend always specifyig output paths)

        - Predicted numpy array: This is used for the model to save its predictions before post processing to get the actual start/end times. This will be saved in a folder called 'Predictions/'. (This might be a redundant operation, more to be investigated)

        - Model start/end elephant predictions: This folder called 'Call_Predictions/' will contain the txt prediction files (selection tables) for each of the recording days!

    Lastly, the data folder with the '.wav' files can exist in any folder. the path just needs to be specified correctly in the comand line argument.

SETUP
    Below are a few basic instructions on how to create an anaconda environment and how 
    to download all of the proper dependancies to run the Inference pipeline script

    Setting up an anaconda environment:

        > conda create --name <name_of_env> --file requirements.txt
        > conda activate <name_of_env>

    If it gives a PackagesNotFoundError, use this command to search packages on more channels.

        > conda config --append channels conda-forge

    Note: Do not include the '>' character when copying the command

    After running these two lines you should be all good to try the example
    runs below!


EXAMPLE RUNS

    Note: It requires python 3.7 and above, preferrably python 3.8

    Generate Predictions:
    python inference.py --make_full_preds --model Model/simple_transformer.pt --data_path <path_to_data_directory> --preds_path <output_path_to_predictions> --call_preds_path <output_path_to_selection_tables> --device <cpu_or_gpu>

    ------------------------------------

    Generate selection table from Predictions:
    python inference.py --save_calls --model Model/simple_transformer.pt --data_path <path_to_data_directory> --preds_path <output_path_to_predictions> --call_preds_path <output_path_to_selection_tables> --device <cpu_or_gpu>

    ------------------------------------

    Full Pipeline:
    python inference.py --make_full_preds --save_calls --model Model/simple_transformer.pt --data_path <path_to_data_directory> --preds_path <output_path_to_predictions> --call_preds_path <output_path_to_selection_tables> --device <cpu_or_gpu>

    ------------------------------------

    Improvement Notes:

    1. The code currently has bad error handling so paths must be set correctly.
    2. Currently the model skips last chunk of audio files if the size is less than 10 seconds.
    3. Add docstrings