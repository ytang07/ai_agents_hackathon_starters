## Getting started
1. Clone the repo and install the required packages using uv(recommended using `uv sync`) or pip
2. Sign up to (LanceDB Cloud)[https://cloud.lancedb.com/].
3. Instruction for applying $50 credit for LanceDB Cloud can be found [on this form](https://docs.google.com/forms/d/e/1FAIpQLScDxl3IlVhL_hlV3lmOB74DDP_tpZZ7y1xdfyKkpx0SXNs4Qw/viewform), please fill the form to get the credit. 
4. Set the GMI Cloud API key and LanceDB project API key env variables.
5. Set LANCEDB_URI and other optional configs in config.py. You can find your project uri from lancedb from the project page, when you generate your API key.
6. Run the ingestion script `python ingest_data_to_lance_cloud.py`
7. Run the app using `streamlit app.py`

### Video Walkthrough

https://github.com/user-attachments/assets/e9670f99-4464-4f8e-a358-80916f3c5bbb

