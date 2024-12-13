from dataclasses import replace
from pathlib import Path
import hexaly.optimizer
from ugraph import NodeIndex

from hackthetrack.dependencygraph.network import DependencyGraph
from hackthetrack.displib.load_displib_instance import DisplibInstance

horizon = 5 * 60 * 60
INFINITE = 1e6


def main():
    displib_instance_path = Path("out/instances/displib_instances_phase1/line2_close_0.json")
    instance = DisplibInstance.from_json(displib_instance_path)

    network = DependencyGraph.from_displib_instance(instance)
    objective_nodes = [
        network.node_by_train_id_and_index(obj["train"], obj["operation"]) for obj in instance.objectives
    ]
    objective_node_indices = [network.node_index_by_name(node.id) for node in objective_nodes]

    num_nodes = len(network.all_nodes)

    node_index_to_train = []
    release_times_by_resource: dict[str, list[float]] = {}

    for i, node in enumerate(network.all_nodes):
        node_index_to_train.append(node.train_id)

        for resource in node.resources:
            if resource.name not in release_times_by_resource:
                release_times_by_resource[resource.name] = [INFINITE for _ in range(network.n_count)]

            release_times_by_resource[resource.name][i] = resource.release_time

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        model = optimizer.get_model()

        operations = [model.interval(node.start_lb if node.start_lb else 0, horizon) for node in network.all_nodes]
        operation_array = model.array(operations)

        resource_allocation = model.array([model.list(num_nodes) for _ in range(len(release_times_by_resource))])

        operation_to_train_array = model.array(node_index_to_train)

        # lower bound on operation duration and upper bound on start time if applicable
        for operation, node in zip(operations, network.all_nodes):
            model.add_constraint(model.length(operation) >= node.min_duration)
            if node.start_ub is not None:
                model.add_constraint(model.start(operation) <= node.start_ub)

        # ordering of operations on the same resource with release times if subsequent operations are on different trains
        for r, release_times_for_resource in enumerate(release_times_by_resource.values()):
            sequence = resource_allocation[r]
            release_times_for_resource_array = model.array(release_times_for_resource)
            sequence_lambda = model.lambda_function(
                lambda i: operation_array[sequence[i]]
                # + (operation_to_train_array[sequence[i]] != operation_to_train_array[sequence[i + 1]])
                # * release_times_for_resource_array[sequence[i]]
                < operation_array[sequence[i + 1]]
            )
            model.constraint(model.and_(model.range(0, model.count(sequence) - 1), sequence_lambda))

        # no-wait and routing of trains
        # ? where is it that a certain train has to use a certain resource? i.e. why can't the resource allocation
        # ? not just be empty?
        for i, node in enumerate(network.all_nodes):

            # only do this for active nodes? definitely not good for an objective which could be inactive
            # maybe we could just do this for all nodes in the sequence and the starting node?
            resources = [r.name for r in node.resources]
            for resource in resources:
                resource_index = list(release_times_by_resource.keys()).index(resource)
                model.add_constraint(model.contains(resource_allocation[resource_index], i))

            neighbors = network.neighbors(node.id, "out")
            if len(neighbors) == 0:
                continue
            elif len(neighbors) == 1:
                model.add_constraint(
                    model.end(operation_array[i])
                    == model.start(operation_array[network.node_index_by_name(neighbors[0].id)])
                )
            else:
                model.add_constraint(
                    model.or_(
                        [
                            model.end(operation_array[i])
                            == model.start(operation_array[network.node_index_by_name(neighbor.id)])
                            for neighbor in neighbors
                        ]
                    )
                )

        # activate all trains

        # objective
        model.minimize(model.sum([model.start(operation_array[i]) for i in objective_node_indices]))
        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = 60 * 10

        optimizer.solve()

        operation_solution = operation_array.value
        for i, node in enumerate(network.all_nodes):
            network.replace_node(
                NodeIndex(i),
                replace(node, start_lb=operation_solution[i].start(), start_ub=operation_solution[i].start()),
            )

        ...


if __name__ == "__main__":
    main()
