import typing as t

import aclick
import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import animation, rc
from sympy import sympify

from .dataset.utils.sympy_functions import evaluate_points, expr_to_func
from .model.runner import Runner, sample_points


def generate_functions(points, eval_points, runner: Runner, *, num_show_equations):
    prediction = runner.predict_all(points=tf.convert_to_tensor([points]))
    prediction = tuple(x[0][:num_show_equations] for x in prediction)
    point_prediction = []
    for prediction_series in prediction[4]:
        point_prediction_series = []
        point_prediction.append(point_prediction_series)
        for symbolic_prediction in prediction_series:
            lam = expr_to_func(symbolic_prediction, runner.variables)
            pred_y = evaluate_points(
                lam, tf.reshape(eval_points, (-1, len(runner.variables)))
            )
            point_prediction_series.append(pred_y)
    max_len = max(len(x) for x in point_prediction)
    point_prediction = [pp + [pp[-1]] * (max_len - len(pp)) for pp in point_prediction]
    point_prediction = np.array(point_prediction)
    return point_prediction[..., -1]


@aclick.command
@click.argument("function", nargs=-1)
def generate_video(
    function: t.List[str],
    /,
    model: str,
    output: t.Optional[str] = None,
    num_show_equations: int = 8,
    num_equations: int = 512,
):
    assert len(function) > 0
    rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    rc("text", usetex=True)
    fig = plt.figure(figsize=(16, 6), dpi=1920 / 16)
    # plt.axis('off')
    # plt.suptitle(r'\textbf{Sym: NeRF-free Neural Rendering from Few Images Using Transformers}', fontsize=26, y=0.85)
    plt.tight_layout()
    # lsequence, index = frames[0]
    # plt.text(0.5, 1.08, subtitle, fontdict={
    #     'fontsize': 26,
    # }, horizontalalignment='center')
    # ctx_text = plt.text(
    #    0.5,
    #    0.98,
    #    r"\textbf{predicting function} ",
    #    fontdict={
    #        "fontsize": 18,
    #    },
    #    horizontalalignment="center",
    # )
    # p1, p2, p3 = fig.subplots(1, 3)
    # p1.imshow(context_images(lsequence, index))
    # p1.set_title('context images', fontsize=22, y=-0.08)
    # p1.axis('off')
    # p2.imshow(imread(f'{lsequence}-gt.png'))
    # p2.set_title('GT', fontsize=22, y=-0.08)
    # p2.axis('off')
    # p3.imshow(imread(f'{lsequence}-gen@{index:02d}.png'))
    # p3.set_title('generated image', fontsize=22, y=-0.08)
    # p3.axis('off')
    # fig.subplots_adjust(bottom=0.1, left=0.05, right=1-0.05, top=1-0.3, wspace=0.05, hspace=0.05)

    runner = Runner.from_checkpoint(model, num_equations=num_equations)

    # Generate all predictions and data to plot later
    functions_data = []
    points_data = []
    predicted_function_data = []

    # mpl.rcParams['axes.spines.left'] = False
    mpl.rcParams["axes.spines.right"] = False
    # mpl.rcParams['axes.spines.bottom'] = False
    mpl.rcParams["axes.spines.top"] = False

    for i, func in enumerate(function):
        minx, maxx = -20, 20
        if "," in func:
            func, minx_str, maxx_str = func.split(",")
            minx = float(minx_str)
            maxx = float(maxx_str)
        lam = expr_to_func(sympify(func), runner.variables)
        points = sample_points(lam, len(runner.variables))
        eval_points = np.linspace(minx, maxx, 1000)
        functions = generate_functions(
            points, eval_points, runner, num_show_equations=num_show_equations
        )
        print("Predicted")

        predicted_function_data.extend([func] * max(50, functions.shape[1]))
        points_data.append(
            np.repeat(
                np.transpose(points)[np.newaxis, ...], max(50, functions.shape[1]), 0
            )
        )
        functions_data.append(np.swapaxes(functions, 0, 1))

        if i == 0:
            plots = [
                plt.plot(eval_points, data, alpha=0.5, color="#1f77b4")[0]
                for data in functions_data[-1][0]
            ]
            scatter_plot = plt.scatter(
                *points_data[-1][0], color="#d96e0f", zorder=3, s=10
            )
    functions_meta = []
    prev = 0
    for data in functions_data:
        functions_meta.append(
            {
                "maxi": np.max(data),
                "mini": np.min(data),
                "num": prev + np.shape(data)[0],
            }
        )
        prev += np.shape(data)[0]
    print(functions_meta)
    functions_data = np.concatenate(functions_data, 0)

    points_data = np.concatenate(points_data, 0)

    plt.axvspan(-5, 5, facecolor="blue", alpha=0.05)

    plt.gcf().text(
        0.28,
        0.9,
        r"SymFormer: End-to-end symbolic regression using transformer-based architecture",
        size=18,
        verticalalignment="center",
    )

    def ani(index):
        for p, data in zip(plots, functions_data[index]):
            p.set_data(eval_points, data)

        current = None
        for meta in functions_meta:
            if index <= meta["num"]:
                current = meta
                break

        plt.xlim(-21, 21)
        mini = current["mini"]
        if current["mini"] < 0:
            mini *= 1.1
        elif current["mini"] > 0:
            mini *= 1 / 1.1
        else:
            mini = 0.9

        maxi = current["maxi"]
        if current["maxi"] < 0:
            maxi *= 1 / 1.1
        elif current["maxi"] > 0:
            maxi *= 1.1
        else:
            maxi = 1.1

        plt.ylim(mini, maxi)
        scatter_plot.set_offsets(np.transpose(points_data[index]))
        plt.title(
            r"\textbf{Predicting function: " + predicted_function_data[index] + "}",
            y=-0.1,
            fontdict={"fontsize": 15},
        )

    animator = animation.FuncAnimation(
        fig, ani, frames=functions_data.shape[0], interval=10, blit=False
    )
    if output is None:
        plt.show()
    else:
        ffmpeg_writer = animation.FFMpegWriter(fps=24)
        animator.save(output, writer=ffmpeg_writer)


if __name__ == "__main__":
    generate_video()
