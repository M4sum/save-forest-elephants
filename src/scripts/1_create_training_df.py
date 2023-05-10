# Creates a merged dataframe containing information about the audio files and when there are rumbles


from src.data.load import get_buckets, load_rumble_clip_data, load_gunshot_clip_data
from src.data.utils import get_s3_client_and_resource, get_training_df_location, create_dirs_for_file
from src.data.preprocess import preprocess_training_df, merge_rumble_and_gunshots

import argparse

# Instantiate the parser
import os
parser = argparse.ArgumentParser(
    description='Creates a single dataframe containing all the information about the sound clips')

parser.add_argument('--dev',
                    action='store_true', help='If we want to use a smaller dev set')

args = parser.parse_args()

# Load the client and the resources
client, s3_resource = get_s3_client_and_resource()

# Create the directories that we need


# Get the buckets from which we want to read the files
s3_bucket = s3_resource.Bucket('data-ai-for-forest-elephants')
text_buckets, wav_buckets = get_buckets(s3_bucket)

print(s3_bucket.objects.all())

gunshot_df = load_gunshot_clip_data(s3_bucket)
rumble_df = load_rumble_clip_data(s3_bucket)



preprocessed_gunshot_df = preprocess_training_df(gunshot_df, wav_buckets)
preprocessed_rumble_df = preprocess_training_df(rumble_df, wav_buckets)


if args.dev:
    preprocessed_gunshot_df = preprocessed_gunshot_df.iloc[:1]
    preprocessed_rumble_df = preprocessed_rumble_df.iloc[:1]



training_df = merge_rumble_and_gunshots(preprocessed_rumble_df, preprocessed_gunshot_df)
save_location = get_training_df_location(args.dev)
create_dirs_for_file(save_location)


training_df.to_csv(save_location, index=False)
