import os
import pickle

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from hcfl.task import load_data, load_model


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
        print(f"Signal received: {signal}")
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


class EdgeClient(NumPyClient):
    def __init__(
        self, model, data, epochs, batch_size, verbose
    ):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(context: Context):
    node_type = context.node_config["type"]

    if node_type == "AGGR":
        return EdgeAggrClient().to_client()

    elif node_type == "WRKR":
        # Load model and data
        model = load_model()

        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]

        data = load_data(partition_id, num_partitions)

        epochs = context.run_config["local-epochs"]
        batch_size = context.run_config["batch-size"]

        verbose = context.run_config.get("verbose")

        # Return Client instance
        return EdgeClient(
            model, data, epochs, batch_size, verbose
        ).to_client()


app = ClientApp(client_fn=client_fn)
