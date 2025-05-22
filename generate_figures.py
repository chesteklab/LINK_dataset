#TODO: BCI DECODING IMPORT
from data_processing.pop_level_analyses import create_dimensionality_across_days_figures
from data_processing.dataset_overview import create_dataset_overview_figure
from data_processing.signal_changes import create_signal_quality_figure
from data_processing.single_channel_tuning import create_single_channel_tuning_figure
from data_processing.bci_decoding import creat_all_decoding_figures
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
        default="C:\\Files\\UM\\ND\\github\\big_nhp_dataset_code\\data\\pickles"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:\\Files\\UM\\ND\\github\\big_nhp_dataset_code\\outputs"
    )
    parser.add_argument(
        "--bci_dir",
        type=str,
        default="C:\\Files\\UM\\ND\\github\\big_nhp_dataset_code\\data_processing\\bci_decoding\\single_day_model_results"
    )

    args = parser.parse_args()
    data_path = args.data_dir
    output_path = args.output_dir
    results_folder = args.bci_dir
    if args.plot is None:
        if data_path is None:
            print("data_dir is required")
            sys.exit(1)
        if output_path is None:
            print("output_dir is required")
            sys.exit(1)
        if results_folder is None: 
            print("No --plot argument and --bci_dir passed. Plotting all figures except BCI figures.")
            creat_all_decoding_figures(results_folder, output_path)
            create_dimensionality_across_days_figures(data_path)
            create_dataset_overview_figure(data_path, output_path)
            create_signal_quality_figure(data_path, output_path, calc_avg_sbp=True, calculate_pr=True)
            create_single_channel_tuning_figure(data_path, output_path)
            raise NotImplementedError("bci_decoding plotting not implemented")
        else:
            print("No --plot argument passed. Plotting all figures.")
            creat_all_decoding_figures(results_folder, output_path)
            create_dimensionality_across_days_figures(data_path)
            create_dataset_overview_figure(data_path, output_path)
            create_signal_quality_figure(data_path, output_path, calc_avg_sbp=True, calculate_pr=True)
            create_single_channel_tuning_figure(data_path, output_path)
            raise NotImplementedError("bci_decoding plotting not implemented")
    if args.plot == "bci_decoding":
        if results_folder is None:
            print("bci_dir is required")
            sys.exit(1)
        if output_path is None:
            print("output_dir is required")
            sys.exit(1)
        creat_all_decoding_figures(results_folder, output_path)
    elif args.plot == "dimensionality_across_days":
        if data_path is None:
            print("data_dir is required")
            sys.exit(1)
        if output_path is None:
            print("output_dir is required")
            sys.exit(1)
        create_dimensionality_across_days_figures(data_path)
    elif args.plot == "dataset_overview":
        if data_path is None:
            print("data_dir is required")
            sys.exit(1)
        if output_path is None:
            print("output_dir is required")
            sys.exit(1)
        create_dataset_overview_figure(data_path, output_path)
    elif args.plot == "signal_quality":
        if data_path is None:
            print("data_dir is required")
            sys.exit(1)
        if output_path is None:
            print("output_dir is required")
            sys.exit(1)
        create_signal_quality_figure(data_path, output_path, calc_avg_sbp=True,calculate_pr=True)
    elif args.plot == "single_channel_tuning":
        if data_path is None:
            print("data_dir is required")
            sys.exit(1)
        if output_path is None:
            print("output_dir is required")
            sys.exit(1)
        create_single_channel_tuning_figure(data_path, output_path)

if __name__ == "__main__":
    parse()