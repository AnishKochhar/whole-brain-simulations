import os
import requests
from tqdm import tqdm
import scipy.io

url = "https://zenodo.org/records/2872624/files/Individual_Connectomes.mat?download=1"
local_filename = "Hagmann_Individual_Connectomes.mat"

if not os.path.exists(local_filename):
    print("Downloading the Hagmann dataset...")

    response = requests.get(url, stream = True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 KB
    with open(local_filename, 'wb') as file, tqdm(
        desc=local_filename,
        total=total_size, 
        unit='iB',
        unit_scale=True
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

    print("Download complete")
else:
    print(f"File '{local_filename}' already exists. Skipping download")

# Load .mat file into memory
print("Loading the .mat file...")
data = scipy.io.loadmat(local_filename)

# Inspect keys
for key in data.keys():
    print(f"KEY: [{key}]")
    print("Type:", data[key].type)