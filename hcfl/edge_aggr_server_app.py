from typing import List, Tuple, Optional, Union

import os
import pickle

from flwr.common import (
    Context,
    Metrics,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy


class EdgeFedAvg(FedAvg):
    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"EdgeFedAvg(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        num_successful_clients = len(results)
        print(f"\n{'-' * 80}\n")
        print(f"[Round {server_round}] Aggregating results from {num_successful_clients} clients.")

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        aggr_weights_filepath = "weights.pkl"
        with open(aggr_weights_filepath, "wb") as file:
            pickle.dump(parameters_aggregated, file)
        print(f"[Round {server_round}] Aggregated weights saved: {aggr_weights_filepath}")

        print("Sending signal...", end='')
        loc_aggr_pipe = "loc_aggr_sig"
        with open(loc_aggr_pipe, "w") as pipe:
            pipe.write("LOC_AGGR_W")  # -> Edge Aggregator client
        print("Done")

        print(f"[Round {server_round}] Aggregation complete.")
        print(f"\n{'-' * 80}\n")

        return parameters_aggregated, metrics_aggregated


# custom (basic) weighted average
def weighted_avg(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    aggregated_metrics: Metrics = {}
    total_examples = 0

    for num_examples, client_metrics in metrics:
        total_examples += num_examples
        for metric_name, metric_value in client_metrics.items():
            aggregated_metrics[metric_name] = aggregated_metrics.get(metric_name, 0.0) + metric_value*num_examples

    for metric_name in aggregated_metrics:
        aggregated_metrics[metric_name] /= total_examples

    return aggregated_metrics


def fit_metrics_aggregation_fn(fit_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    print("Aggregating fit metrics from workers:")
    for i, (_, per_client_metrics) in enumerate(fit_metrics):
        print(f"  ({i}) {per_client_metrics.items()}")
    aggregated_metrics = weighted_avg(fit_metrics)
    return aggregated_metrics


def evaluate_metrics_aggregation_fn(eval_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    print("Aggregating evaluate metrics from workers:")
    for i, (_, per_client_metrics) in enumerate(eval_metrics):
        print(f"  ({i}) {per_client_metrics.items()}")
    aggregated_metrics = weighted_avg(eval_metrics)
    return aggregated_metrics


def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = context.run_config["num-server-rounds"]

    print("Waiting for global aggregation to finish...")

    glb_aggr_pipe = "glb_aggr_sig"
    if not os.path.exists(glb_aggr_pipe):
        os.mkfifo(glb_aggr_pipe)

    with open(glb_aggr_pipe, "r") as pipe:
        signal = pipe.readline().strip()  # wait for GLB_AGGR_W from Edge Aggregator server
    print(f"Signal received: {signal}\n")
    os.remove(glb_aggr_pipe)

    gbl_weights_filepath = "weights.pkl"
    print(f"Loading weights from file: {gbl_weights_filepath}... ", end='')
    with open(gbl_weights_filepath, "rb") as file:
        parameters_ndarrays = pickle.load(file)
    print("Done")
    parameters = ndarrays_to_parameters(parameters_ndarrays)
    print(type(parameters))

    print("Removing weights file... ", end='')
    os.remove(gbl_weights_filepath)
    print("Done")

    strategy = EdgeFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
