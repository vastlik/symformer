from aclick import group

from .dataset.dataset import generate_dataset_command as generate_dataset_command

from .dataset.visualize_dataset import main as visualize_dataset
from .evaluate import evaluate
from .evaluate_benchmark import evaluate as evaluate_benchmark
from .generate_video import generate_video
from .predict import main as predict
from .train import train as train


@group
def main():
    pass


main.add_command(train)
main.add_command(visualize_dataset)
main.add_command(evaluate_benchmark)
main.add_command(evaluate)
main.add_command(predict)
main.add_command(generate_video)
main.add_command(generate_dataset_command)

if __name__ == "__main__":
    main()
