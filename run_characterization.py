import config
import matplotlib.pyplot as plt
import os
from dataset_characterization import dataset_characterization


if __name__=="__main__":
    dates =  dataset_characterization.get_good_days()
    # create_time_plot(dates)

    fig, ax = plt.subplots(1, 2, figsize = (19.5, 6), sharex=True)
    dataset_characterization.signal_power_over_time(dates, ax[0], crunch=False)
    # dataset_characterization.calc_participation_ratios(dates)
    dataset_characterization.create_pr_plot(ax[1])

    plt.savefig(os.path.join(config.characterizationdir, "sbp_and_PR_over_time.pdf"))
    plt.show()

    # dataset_characterization.sfn_decoding()