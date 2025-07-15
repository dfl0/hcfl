import os
import numpy as np

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays


class EdgeAggrClient(NumPyClient):

    def fit(self, parameters, config):
        filepath = "weights.npz"
        print(f"saving global model weights to file: {filepath}")
        np.savez(filepath, *parameters)

        print(f"global model weights saved to {filepath}")

        print("sending signal")
        glb_aggr_pipe = "glb_aggr_sig"
        with open(glb_aggr_pipe, "w") as pipe:
            pipe.write("GLB_AGGR_W")  # -> Edge Aggregator server
        print("done")

        print("waiting for local training and aggregation to finish...")

        loc_aggr_pipe = "loc_aggr_sig"
        if not os.path.exists(loc_aggr_pipe):
            os.mkfifo(loc_aggr_pipe)

        with open(loc_aggr_pipe, "r") as pipe:
            signal = pipe.readline().strip()  # wait for LOC_AGGR_W from Edge Aggregator server
        print(f"signal received: {signal}\n")
        os.remove(loc_aggr_pipe)

        print("loading aggregated weights from file")
        aggr_weights_ndarrays = np.load("weights.npz")
        print("done")

        print("removing weights file")
        os.remove("weights.npz")
        print("done")

        return aggr_weights_ndarrays, 1, {}

    def evaluate(self, parameters, config):
        return 1.0, 1, {"accuracy": 0.0}


def client_fn(context: Context):
    return EdgeAggrClient().to_client()


app = ClientApp(client_fn=client_fn)
