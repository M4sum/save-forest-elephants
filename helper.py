import asyncio
import numpy as np
import os

def load_spectrograms(filename, file_id):
    with open(filename, 'rb') as f:
        spectrogram = np.load(f)
    return spectrogram, file_id

def save_predictions(model_id, data_id, hierarchical_model, hierarchical_predictions, predictions, predictions_path):
# Save preditions
        # Save for now to a folder determined by the model id
        path = os.path.join(predictions_path,model_id)
        if not os.path.isdir(path):
            os.mkdir(path)

        if hierarchical_model is not None:
            # Create a folder for the model 0 predictions
            model_0_path = os.path.join(path, "Model_0")
            if not os.path.isdir(model_0_path):
                os.mkdir(model_0_path)

            np.save(os.path.join(model_0_path, data_id + '.npy'), predictions)
            np.save(os.path.join(path, data_id + '.npy'), hierarchical_predictions)
        else:
            # The data id associates predictions with a particular spectrogram
            np.save(os.path.join(path, data_id  + '.npy'), predictions)