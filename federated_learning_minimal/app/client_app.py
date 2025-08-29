"""HighVariance: A Flower for high-variance genes identification."""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import numpy as np

from app.task import get_random_integer, load_local_data

# Define Flower Client
class FlowerClient(NumPyClient):

    # Initilize Flower Client
    def __init__(self, 
        data,
    ):
        self.data = data

    # Get gene variances
    def fit(self, parameters, config):    
        gene_variances = np.var(self.data.layers["counts"], axis=0)
        weight = self.data.shape[0]

        return [gene_variances], weight, {}  

def client_fn(context: Context):

    # Retrieve the adata file path from the context
    adata_file_path = context.run_config["adata_file_path"]

    # Get the node id
    partition_id = context.node_config["partition-id"]

    # Load the local data for the client
    adata_local = load_local_data(partition_id, adata_file_path)
    print(f"Client ID: {partition_id} | Local data shape: {adata_local.shape}")
    
    return FlowerClient(adata_local).to_client() 

# Flower ClientApp
app = ClientApp(client_fn=client_fn)