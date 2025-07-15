from typing import List, Tuple, Optional, Union

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

from hcfl.task import load_model


class CentralFedAvg(FedAvg):
    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"CentralFedAvg(accept_failures={self.accept_failures})"
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
            # Accumulate the weighted sum for each metric
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
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Get parameters to initialize global model
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # Define strategy
    strategy = CentralFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
