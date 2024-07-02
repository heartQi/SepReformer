import argparse
import importlib

# Parse args
parser = argparse.ArgumentParser(
    description="Command to start PIT training, configured by .yaml files")
parser.add_argument(
    "--model",
    type=str,
    default="IQformer_v6_0_0",
    dest="model",
    help="Insert model name")
parser.add_argument(
    "--engine_mode",
    choices=["train", "test", "test_wav"],
    default="train",
    help="This option is used to chooose the mode")
parser.add_argument(
    "--out_wav_dir",
    type=str,
    default=None,
    help="This option is used to specficy save directory for output wav file in test_wav mode")
parser.add_argument(
    "--non_chunk",
    type=int,
    default=1,
    help="This option for chunk")
parser.add_argument(
    "--chunk_size",
    type=int,
    default=1600,
    help="This option for chunk")

args = parser.parse_args()

# Call target model
main_module = importlib.import_module(f"models.{args.model}.mainLocal")
main_module.main(args)