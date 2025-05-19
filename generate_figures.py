#TODO: BCI DECODING IMPORT
from analysis.pop_level_analyses import create_dimensionality_across_days_figures
from analysis.dataset_overview import create_dataset_overview_figure
from analysis.signal_changes import create_signal_quality_figure
from analysis.single_channel_tuning import create_single_channel_tuning_figure
import argparse
import sys

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot",
        choices=[
            "bci_decoding",
            "dimensionality_across_days",
            "dataset_overview",
            "signal_quality",
            "single_channel_tuning"
        ],
        required=False
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="C:\\Files\\UM\\ND\\SFN\\only_good_days"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:\\Files\\UM\\ND\\github\\big_nhp_dataset_code\\outputs"
    )

    args = parser.parse_args()
    data_path = args.data_dir
    output_path = args.output_dir
    if args.plot is None:
        print("No --plot argument passed. Plotting all figures.")
        # TODO: plot bci figures
        create_dimensionality_across_days_figures(data_path)
        create_dataset_overview_figure(data_path, output_path)
        create_signal_quality_figure(data_path, output_path, calc_avg_sbp=True, calculate_pr=True)
        create_single_channel_tuning_figure(data_path, output_path)
        raise NotImplementedError("bci_decoding plotting not implemented")
    if args.plot == "bci_decoding":
        #TODO: plot bci figures
        raise NotImplementedError("not implemented")
    elif args.plot == "dimensionality_across_days":
        create_dimensionality_across_days_figures(data_path)
    elif args.plot == "dataset_overview":
        create_dataset_overview_figure(data_path, output_path)
    elif args.plot == "signal_quality":
        create_signal_quality_figure(data_path, output_path, calc_avg_sbp=True,calculate_pr=True)
    elif args.plot == "single_channel_tuning":
        create_single_channel_tuning_figure(data_path, output_path)

if __name__ == "__main__":
    parse()