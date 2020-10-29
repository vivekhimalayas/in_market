
from os import listdir, makedirs
import os
from os.path import join, isdir, isfile, basename
import datetime
from google.cloud import bigquery, storage
import numpy as np
import pandas as pd
import keras

from keras.layers import Input, Dense ,Lambda , Dropout, concatenate, Activation, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K
import shutil




storage_client = storage.Client(project="sharechat-production")
bq_client = bigquery.Client(project="maximal-furnace-783")
bucket = storage_client.get_bucket("daily-aggregation-results")
blobs = list(bucket.list_blobs(prefix= "ad_relevance/in_market_explicit/"))
if blobs:
            # GCS folder already exists. Deleting it's content
            for blob in blobs:
                blob.delete()
bucket_name = "daily-aggregation-results"
folder_name = "ad_relevance/in_market_explicit/"
file_extention = "csv"
destination_format="CSV"
destination_uri = 'gs://{}/{}/*.{}'.format(bucket_name, folder_name, file_extention)
extraction_table = bq_client.dataset("churn_analysis").table("imc_input_data")

job_config = bigquery.ExtractJobConfig()
job_config.destination_format = destination_format
job_config.compression = "None"
job_config.field_delimiter = ','
extract_job = bq_client.extract_table(extraction_table, destination_uri, job_config=job_config)
extract_job.result()

# uncomment the line below if you have a service account json
# storage_client = storage.Client.from_service_account_json('creds/sa.json')

dest_path = "Data/in_market_explicit/"

if not os.path.exists(dest_path):
            os.makedirs(dest_path)
else:
    shutil.rmtree(dest_path)
    os.makedirs(dest_path)

blobs = bucket.list_blobs(prefix=folder_name)  # Get list of files
for blob in blobs:
    blob_name = blob.name
    dst_file_name = blob_name.replace("ad_relevance/in_market_explicit/", dest_path)
    dst_dir = dst_file_name.replace('/' + basename(dst_file_name), '')
    if isdir(dst_dir) == False:
        makedirs(dst_dir)
    # download the blob object
    blob.download_to_filename(dst_file_name)



print("File_downloaded in the local updated")


model = keras.models.load_model("in_market_catg_prediction_v1")

print("model loading done")



train_dir = "Data/in_market_explicit/"

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]
filename = []

filenames = find_csv_filenames(train_dir)
for name in filenames:
      filename.append(name)
print("Number of files")  
print(len(filename))      
        
def test_generate_batches(files, batch_size):
   global userid 
   userid = pd.DataFrame()
   counter = 0
   while counter < len(files):
     fname = files[counter]
     counter = (counter + 1)
     data_bundle = pd.read_csv(train_dir + fname)
     X_train = data_bundle.iloc[:,1:]
     userid = pd.concat([userid,data_bundle.iloc[:,0]])
     for cbatch in range(0, X_train.shape[0], batch_size):
         
         yield (np.array(X_train.iloc[cbatch:(cbatch + batch_size),:]))

test_files = filename
test_gen = test_generate_batches(files=test_files, batch_size=8192)



print("prediction starting")

pred =  model.predict(test_gen)

print("prediction done")

columns = ["Shopping","Social","Books","Card","News","Entertainment","Photography","Video_players","Education","Finance","Lifestyle","Sports",
        "Tools","Business","Music","Communication","Health","Puzzle","Casual","Board","Productivity","Action","Travel","Casino","Medical","Auto_vehicle",
        "Food_Drink","Maps","Personalisation","Arcade","Simulation","Racing","Dating","Strategy","Parenting","Beauty","Role_playing","House_home",
        "Events","Adventure","Word","Weather","Trivia","Art_Design","Comics","Educational"]


prediction = pd.DataFrame()
prediction["userid"] = userid.iloc[:,0].astype(int)
i = 0
for x in columns:
  prediction[x] = pred[:,i]
  i = i+1
print("Prediction datafram created for dumping in BQ")

import gc
gc.collect()

client = bigquery.Client(project="maximal-furnace-783")

# TODO(developer): Set table_id to the ID of the table to create.
# table_id = "your-project.yourdataset.your_table_name"
prediction.to_csv("model_score.csv", index = False)
job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1, autodetect=True,
    write_disposition=bigquery.job.WriteDisposition.WRITE_TRUNCATE
)
table_id =  'churn_analysis.in_market_score'

with open("model_score.csv", "rb") as source_file:
    job = client.load_table_from_file(source_file, table_id, job_config=job_config)

job.result() 

table = client.get_table(table_id)  # Make an API request.
print(
    "Loaded {} rows and {} columns to {}".format(
        table.num_rows, len(table.schema), table_id
    )
)







