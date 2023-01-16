#Instructions

- We need to have conda installed
- First we need to install mamba (conda package) using the following command
  conda install mamba -n base -c conda-forge
- Clone the repo and enter into the directory
  git clone https://github.com/texsmv/airq_tool
  cd airq_tool
- Download the datasets from drive and uncompress them into the root directory
- We now need install the required packages using mamba
  mamba env create --file clearn.yaml
- Activate the environment
  mamba activate clearn
- Run the python server
  python app_server.py
- Run the server for the web application (On a separate terminal)
  mamba activate clearn
  cd web
  python -m http.server
- In a web browser open the following (http://0.0.0.0:8000), the first time it may take some time to load
- THATS ALL!
