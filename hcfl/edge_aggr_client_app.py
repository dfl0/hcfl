import os
import pickle

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context


class EdgeAggrClient(NumPyClient):

    def fit(self, parameters, config):
        global_weights_filepath = "weights.pkl"
        with open(global_weights_filepath, "wb") as file:
            pickle.dump(parameters, file)
        print(f"Global model weights saved: {global_weights_filepath}")

        print("Sending signal...", end='')
        glb_aggr_pipe = "glb_aggr_sig"
        with open(glb_aggr_pipe, "w") as pipe:
            pipe.write("GLB_AGGR_W")  # -> Edge Aggregator server
        print("Done")

        print("Waiting for local training and aggregation to finish...")

        loc_aggr_pipe = "loc_aggr_sig"
        if not os.path.exists(loc_aggr_pipe):
            os.mkfifo(loc_aggr_pipe)

        with open(loc_aggr_pipe, "r") as pipe:
            signal = pipe.readline().strip()  # wait for LOC_AGGR_W from Edge Aggregator server
        print(f"Signal received: {signal}\n")
        os.remove(loc_aggr_pipe)

        updated_weights_filepath = "weights.pkl"
        print(f"Loading weights from file: {updated_weights_filepath}...", end='')
        with open(updated_weights_filepath, "rb") as file:
            updated_parameters_ndarrays = pickle.load(file)
        print("Done")

        print("Removing weights file... ", end='')
        os.remove(updated_weights_filepath)
        print("Done")

        return updated_parameters_ndarrays, 1, {}

    def evaluate(self, parameters, config):
        return 1.0, 1, {"accuracy": 0.0}  # TODO: replace placeholder return data with actual evaulation logic


def client_fn(context: Context):
    return EdgeAggrClient().to_client()


app = ClientApp(client_fn=client_fn)
