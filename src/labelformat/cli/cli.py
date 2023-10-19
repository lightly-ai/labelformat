import logging
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from pathlib import Path
from typing import Optional

from labelformat.cli.registry import _REGISTRY, Task
from labelformat.formats import *

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def main() -> None:
    parser = ArgumentParser(prog="labelformat", description="Labelformat")
    subparsers = parser.add_subparsers(dest="command")
    convert_parser = subparsers.add_parser(
        name="convert",
        description="Convert labels from one format to another.",
        add_help=False,
    )
    convert_parser.add_argument("-h", "--help", action="store_true")
    # Parse already to check if --help flag is set. We have to do this now because
    # the convert --input-format and --output-format arguments are required and
    # raise an error otherwise.
    args, remaining_args = parser.parse_known_args()

    if args.command == "convert":
        convert_parser.add_argument(
            "--task",
            choices=sorted({task.value for task in Task}),
            type=str,
            required=True,
        )

        # Parse the task argument. We modify the input/output format choices
        # based on it. We print the help message first if the task argument is missing.
        if args.help and "--task" not in remaining_args:
            convert_parser.print_help()
            return
        args, _ = parser.parse_known_args()
        task = Task(args.task)

        # Add input/output format arguments.
        convert_parser.add_argument(
            "--input-format",
            choices=sorted(
                set(
                    name
                    for tsk, name_to_cls in _REGISTRY.input.items()
                    for name in name_to_cls
                    if tsk == task
                )
            ),
            help="Input format",
            required=True,
        )
        convert_parser.add_argument(
            "--output-format",
            choices=sorted(
                set(
                    name
                    for tsk, name_to_cls in _REGISTRY.input.items()
                    for name in name_to_cls
                    if tsk == task
                )
            ),
            type=str,
            help="Output format",
            required=True,
        )

        # Print help message if input/output format arguments are missing. If both
        # arguments are set, then we want to wait with printing the help message
        # until the input/output format specific arguments are added to the parser.
        if args.help and (
            "--input-format" not in remaining_args
            or "--output-format" not in remaining_args
        ):
            convert_parser.print_help()
            return

        # Parse again to verify that all arguments are set.
        # This raises an error if --input-format or --output-format is missing.
        args, _ = parser.parse_known_args()

        # Make groups for input and output arguments. This groups input/output arguments
        # in the help message.
        input_group = convert_parser.add_argument_group(
            f"'{args.input_format}' input arguments"
        )
        output_group = convert_parser.add_argument_group(
            f"'{args.output_format}' output arguments"
        )

        # Add arguments depending on input/output format. We use a new parser here
        # to make typing for users easier when defining the add_parser_arguments in
        # the input/output format classes. This allows users to import
        # 'argparse.ArgumentParser' instead of 'argparse._ArgumentGroup'.
        input_parser = ArgumentParser(add_help=False)
        output_parser = ArgumentParser(add_help=False)
        task = Task(args.task)
        input_cls = _REGISTRY.input[task][args.input_format]
        output_cls = _REGISTRY.output[task][args.output_format]
        input_cls.add_cli_arguments(parser=input_parser)
        output_cls.add_cli_arguments(parser=output_parser)

        # Copy added arguments to the input/output group.
        for action in input_parser._actions:
            input_group._add_action(action)
        for action in output_parser._actions:
            output_group._add_action(action)

        # Dispaly help message if requested. This help message will now also contain
        # the input/output format specific arguments.
        if args.help:
            convert_parser.print_help()
            return

        # Parse and get group specific arguments.
        args = parser.parse_args()
        input_args = _get_group_args(args, input_group)
        output_args = _get_group_args(args, output_group)

        # Create input and output instances and convert.
        logger.info("Loading labels...")
        label_input = input_cls(**vars(input_args))
        label_output = output_cls(**vars(output_args))

        logger.info("Converting labels...")
        label_output.save(label_input=label_input)

        logger.info("Done!")
    else:
        # Print a help message if no command is given.
        # Because only convert command is currently available we print its help message.
        convert_parser.print_help()
        return


def _get_group_args(args: Namespace, group: _ArgumentGroup) -> Namespace:
    return Namespace(
        **{
            action.dest: getattr(args, action.dest, None)
            for action in group._group_actions
        }
    )
