import argparse
import json
import math
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

import mnist

app = Flask(__name__)


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = nn.relu(l(x))
        return self.layers[-1](x)


def loss_fn(model, X, y):
    return nn.losses.cross_entropy(model(X), y, reduction="mean")


def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s: s + batch_size]
        yield X[ids], y[ids]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples")
def samples():
    dset = request.args.get("dataset", "mnist")
    train_images, train_labels, _, _ = map(mx.array, getattr(mnist, dset)())
    rng = np.random.default_rng(42)
    indices = rng.choice(len(train_images), 25, replace=False)
    result = []
    for idx in indices:
        img = np.array(train_images[idx]).reshape(28, 28).tolist()
        result.append({"image": img, "label": int(train_labels[idx])})
    return jsonify(result)


DATASET_SIZES = {"mnist": 60000, "fashion_mnist": 60000}


@app.route("/api/dataset-info")
def dataset_info():
    dset = request.args.get("dataset", "mnist")
    train_images, train_labels, test_images, test_labels = map(
        mx.array, getattr(mnist, dset)()
    )
    return jsonify({
        "name": dset,
        "train_size": len(train_images),
        "test_size": len(test_images),
        "image_dim": train_images.shape[-1],
    })


def calc_params(hidden_dim, num_layers):
    sizes = [784] + [hidden_dim] * num_layers + [10]
    total = 0
    for i in range(len(sizes) - 1):
        total += sizes[i] * sizes[i + 1] + sizes[i + 1]
    return total


@app.route("/api/param-info")
def param_info():
    hidden_dim = int(request.args.get("hidden_dim", 32))
    num_layers = int(request.args.get("num_layers", 2))
    return jsonify({
        "total_params": calc_params(hidden_dim, num_layers),
        "layer_sizes": [784] + [hidden_dim] * num_layers + [10],
    })


@app.route("/api/train/sse")
def train_sse():
    p = lambda key, default, cast: cast(request.args.get(key, default))
    params = {
        "hidden_dim": p("hidden_dim", 32, int),
        "num_layers": p("num_layers", 2, int),
        "num_epochs": p("num_epochs", 10, int),
        "batch_size": p("batch_size", 256, int),
        "learning_rate": p("learning_rate", 0.1, float),
        "optimizer": p("optimizer", "sgd", str),
        "dataset": p("dataset", "mnist", str),
        "seed": p("seed", 0, int),
        "run_id": p("run_id", "run_0", str),
    }
    params["total_params"] = calc_params(params["hidden_dim"], params["num_layers"])
    params["layer_sizes"] = [784] + [params["hidden_dim"]] * params["num_layers"] + [10]

    def generate():
        hp = params
        np.random.seed(hp["seed"])

        train_images, train_labels, test_images, test_labels = map(
            mx.array, getattr(mnist, hp["dataset"])()
        )

        model = MLP(hp["num_layers"], 784, hp["hidden_dim"], 10)
        mx.eval(model.parameters())

        if hp["optimizer"] == "adam":
            optimizer = optim.Adam(learning_rate=hp["learning_rate"])
        else:
            optimizer = optim.SGD(learning_rate=hp["learning_rate"])

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

        if hp["optimizer"] == "adam":
            def step(X, y):
                loss, grads = loss_and_grad_fn(model, X, y)
                optimizer.update(model, grads)
                mx.eval(loss, model.state)
                return loss
        else:
            @partial(mx.compile, inputs=model.state, outputs=model.state)
            def step(X, y):
                loss, grads = loss_and_grad_fn(model, X, y)
                optimizer.update(model, grads)
                return loss

        @partial(mx.compile, inputs=model.state)
        def eval_fn(X, y):
            return mx.mean(mx.argmax(model(X), axis=1) == y)

        total_batches = math.ceil(len(train_images) / hp["batch_size"])
        yield f"data: {json.dumps({'type': 'start', 'params': hp, 'total_batches': total_batches})}\n\n"

        for e in range(hp["num_epochs"]):
            epoch_loss = 0.0
            batch_count = 0
            tic = time.perf_counter()
            for X, y in batch_iterate(hp["batch_size"], train_images, train_labels):
                loss_val = step(X, y)
                mx.eval(model.state)
                l = float(loss_val.item())
                epoch_loss += l
                batch_count += 1
                yield f"data: {json.dumps({'type': 'batch', 'epoch': e, 'batch': batch_count, 'total_batches': total_batches, 'loss': round(l, 4)})}\n\n"

            accuracy = float(eval_fn(test_images, test_labels).item())
            toc = time.perf_counter()
            yield f"data: {json.dumps({'type': 'epoch', 'epoch': e, 'accuracy': round(accuracy, 4), 'avg_loss': round(epoch_loss / batch_count, 4), 'time': round(toc - tic, 3)})}\n\n"

        rng = np.random.default_rng(42)
        pred_indices = rng.choice(len(test_images), 20, replace=False)
        predictions = []
        for idx in pred_indices:
            img = test_images[idx]
            predictions.append({
                "image": np.array(img).reshape(28, 28).tolist(),
                "true_label": int(test_labels[idx]),
                "predicted": int(mx.argmax(model(mx.array([img])), axis=1)[0]),
            })

        yield f"data: {json.dumps({'type': 'done', 'predictions': predictions})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MNIST Web UI")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if not args.gpu:
        mx.set_default_device(mx.cpu)
    app.run(debug=True, port=args.port)
