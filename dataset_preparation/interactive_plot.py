import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output, update_display
import pickle
import os
import pandas as pd
import pdb
import matplotlib.cm as cm

class neural_plot(widgets.VBox):
    """
    Interactive widget for visualizing preprocessed neural data.
    """
    def __init__(self, data_path, output_path):
        super().__init__()
        self.datapath = data_path
        self.outputpath = output_path
        self.filenames = [f for f in os.listdir(self.datapath) if not f.startswith('.')] #in case there are any hidden files
        #self.filenames.remove('bad_days.txt')
        self.filenames.sort() #for some reason the files weren't in chronological order if left unsorted, maybe a macOS problem?
        self.fileidx = 0

        self.data = None
        self.plot_start_index = 0
        self.end_index = 0
        self.data_CO = None
        self.data_RD = None
        self.prev = False
        self.current_TS = 'CO'
        self.both_TS_present = False
        self.trial_num_printed = False # added this to stop the keep printing the number of trials
        self.trial_lines = [] # added this to keep track of the vertical lines to avoid duplication
        self.trial_texts = [] # added this to keep track of the text annotations for each trial start
        
        self.timerange = 500
        self.figoutput = widgets.Output()
        self.results_df = pd.DataFrame(columns=['Date','Status','Note'])

        with self.figoutput:
            self.fig, self.ax = plt.subplots(4, 1, figsize= (8,8), constrained_layout=True)

            self.ax[0].set_title('Neural (Unsmoothed, red trace = average over chans)')
            self.ax[0].set_ylabel('Normalized Binned SBP')
            self.ax[0].set_ylim([-3,3])

            self.ax[1].set_title('Threshold Crossings(TCFR)')
            self.ax[1].set_ylabel('TCFR')
            self.ax[1].set_ylim([-5,5])

            self.ax[2].set_title('Finger Position (blue = index, orange = MRL)')
            self.ax[2].set_ylabel('Proportion of Flexion')
            self.ax[2].set_ylim([-0.1,1.1])

            self.ax[3].set_title('Finger Velocity')
            self.ax[3].set_ylabel('Flexion per .01s')
            self.ax[3].set_xlabel('Time (seconds)')
            self.ax[3].set_ylim([-0.125, 0.125])


        # Init neural lines
        cmap = cm.get_cmap('viridis', 96) 
        self.neural_data = []
        for i in range(96):
            line, = self.ax[0].plot(np.arange(100), np.zeros((100,1)), linewidth=0.4, color=[np.random.rand(), np.random.rand(), np.random.rand()])
            self.neural_data.append(line)
        self.average_line, = self.ax[0].plot([], [], color='red', linewidth=1.5)
        self.ax[0].set(xlabel='Time (seconds)', ylabel='Binned SBP', title='Neural (Unsmoothed, red trace = average over chans)')

        # Init TCFR plots
        self.tcfr_data = []
        for i in range(96):
            line, = self.ax[1].plot(np.arange(100), np.zeros((100, 1)), linewidth=0.4, color=cmap(i))
            self.tcfr_data.append(line)
        self.tcfr_avg_line, = self.ax[1].plot(np.arange(100), np.zeros((100,1)), linewidth=0.4,color=cmap(i))
        self.ax[1].set(xlabel='Time (seconds)', ylabel='TCFR', title='Threshold Crossings (TCFR)', ylim=(-5,5))
        
        # Init Finger lines
        self.finger_positions = []
        self.finger_velocities = []
        clist = ['b','orange','g','r']
        for i in range(2):
            line_pos, = self.ax[2].plot([], [], color=clist[i])
            self.finger_positions.append(line_pos)
            line_vel, = self.ax[3].plot([], [], color=clist[i+2])
            self.finger_velocities.append(line_vel)
            
        self.ax[2].set(xlabel='Time (s)', ylabel='Proportion of Flexion', title='Finger Position (blue = index, orange = MRL)', ylim=(-0.1,1.1))
        self.ax[3].set(xlabel='Time (s)', ylabel='Proportion of Flexion \nper .01s', title='Finger Velocity', ylim=(-.125,.125))

        # DEFINE WIDGETS

        # Terminate button
        terminate = widgets.Button(description='Terminate Plotting', button_style='danger')
        terminate.on_click(self.save_and_quit)

        # Switch between target styles (if present)
        switch_styles = widgets.Button(description='Change Target Style', button_style='success')
        switch_styles.on_click(self.switch_TS)

        # Next/Previous day buttons
        next_day = widgets.Button(description='NEXT DAY & HOLD', button_style = 'info')
        next_day.on_click(self.next_day)

        prev_day = widgets.Button(description='PREV DAY & HOLD')
        prev_day.on_click(self.prev_day)

        # Time Scrubbers (500 bins)
        prev_time = widgets.Button(description='← (-500 bins)') 
        prev_time.on_click(self.shift_back)
        next_time = widgets.Button(description='→ (+500 bins)')
        next_time.on_click(self.shift_forward)

        # Good/Bad Run Selection Buttons
        self.good_bad = widgets.RadioButtons(options=[('good', 'good'), ('bad', 'bad')], 
                                        description='Run Status:', 
                                        disabled=False)
        
        # Note Text Box
        self.notes = widgets.Textarea(value='', 
                         placeholder='Enter note for the day (optional)', 
                         description='Note:', 
                         disabled=False)

        hbox1 = widgets.HBox([prev_time, next_time, switch_styles])
        hbox2 = widgets.HBox([self.good_bad, self.notes])
        hbox3 = widgets.HBox([prev_day, next_day, terminate])
        self.text_output = widgets.Output()
        controls = widgets.VBox([hbox1, hbox2, hbox3, self.text_output])
        out_box = widgets.Box([self.figoutput])

        # style the borders/padding
        self.figoutput.layout = widgets.Layout(
            border='solid 1px black',
            margin='0px 10px 10px 0px',
            padding='5px 5px 5px 5px'
            )
        controls.layout = widgets.Layout(
            border='solid 1px black',
            margin='0px 10px 10px 0px',
            padding='5px 5px 5px 5px'
            )
        

        with self.text_output:
            print('Plot Initialized')

        self.children = [self.figoutput, controls]
        self.displayhandle = display(self)

    def start(self, resume, start_from_date = None):
        
        # start from first day or resume based on save file
        if resume and start_from_date:
            if start_from_date:
                #look for specific date
                date_found = False
                for i in range(len(self.filenames)):
                    if self.filenames[i][0:10] == start_from_date:
                        self.fileidx = i
                        date_found = True
                if date_found == False:
                    with self.text_output:
                        print('Date not found, resuming from save state/default')
                else:
                    with self.text_output:
                        print('Date found, resuming')
        elif resume:
            with open(self.outputpath, 'rb') as f:
                self.fileidx = pickle.load(f)
            with self.text_output:
                print(f'Resuming from {self.filenames[self.fileidx]}')
        else:
            with self.text_output:
                print(f'Starting at {self.filenames[self.fileidx]}')

        self.load_current_day()
        self.plot_day()
        return

    def load_current_day(self):

        with open(os.path.join(self.datapath, self.filenames[self.fileidx]), 'rb') as f:
            self.data_CO, self.data_RD = pickle.load(f) 

            
        if (self.data_CO is not None) and (self.data_RD is not None):
            self.both_TS_present = True
            self.current_TS = 'CO' # by default show center out first
            with self.text_output:
                print(f'loaded {self.filenames[self.fileidx]}, both target styles available')
                print(f'Target Style = {self.current_TS}') # make it print the current TS
        else:
            self.both_TS_present = False
            self.current_TS = 'CO' if self.data_RD is None else 'RD'
            with self.text_output:
                print(f'loaded {self.filenames[self.fileidx]}, {self.current_TS}')

        self.plot_start_index = 0
    
    def load_adj_day(self):
        self.trial_num_printed = False
        with self.text_output:
            clear_output()
            print(f'Loading {"prev" if self.prev else "next"} Day...')

        if self.prev:
            self.fileidx -= 1
        else:
            self.fileidx += 1
        
        if self.fileidx < 0 or self.fileidx >= len(self.filenames):
            with self.text_output:
                print(f'END REACHED, STAYING AT CURRENT DAY {self.filenames[self.fileidx]}')
            return

        self.load_current_day()
        self.plot_day()

    def next_day(self, b):
        self.prev = False
        self.save_day()
        self.load_adj_day()
        return

    def prev_day(self, b):
        self.prev = True
        self.save_day()
        self.load_adj_day()
        return
    
    def plot_day(self):
        #plot the current data starting at the first set of 5 trials
        self.Data = self.data_CO if self.current_TS == 'CO' else self.data_RD
        num_trials = len(self.Data['trial_index'])
        if not self.trial_num_printed: # only prints the number of trials when the day/TS is first loaded
            with self.text_output:
                print(f'{num_trials} trials available')
            self.trial_num_printed = True
        
        if self.plot_start_index < 0:
            self.plot_start_index = 0
            with self.text_output:
                print('cannot move plot further back')

        self.end_index = self.plot_start_index + self.timerange
        if self.end_index > len(self.Data['time']):
            self.end_index = len(self.Data['time'])
            with self.text_output:
                print('cannot move plot further forward')
        
        time_slice = slice(self.plot_start_index, self.end_index)
        exp_time = self.Data['time'][time_slice] / 1000 # put in sec
        time_lims = (exp_time[0],exp_time[-1])

        # update neural data
        for i, line in enumerate(self.neural_data):
            line.set_data(exp_time,self.Data['sbp'][time_slice,i])
        self.average_line.set_data(exp_time,np.mean(self.Data['sbp'],axis=1)[time_slice])
        self.ax[0].set(xlabel=None, ylabel='Normalized Binned SBP', title='Neural (Unsmoothed + Average (Red))', xlim=time_lims,ylim=(-5,5))

        # update TCFR data
        for i, line in enumerate(self.tcfr_data):
            line.set_data(exp_time, self.Data['tcfr'][time_slice, i])
        self.tcfr_avg_line.set_data(exp_time, np.mean(self.Data['tcfr'],axis=1)[time_slice])
        self.ax[1].set(xlabel=None, ylabel='TCFR', title='Threshold Crossings (TCFR)', xlim=time_lims, ylim=(-5,5))

        # update finger positions
        for i, line_pos in enumerate(self.finger_positions):
            line_pos.set_data(exp_time, self.Data['finger_kinematics'][time_slice,i])
        self.ax[2].set(xlabel=None, ylabel='Flexion', title='Finger Positions', xlim=time_lims,ylim=(-0.1,1.1)) # changed ylim a bit

        
        # update finger velocities
        for i, line_vel in enumerate(self.finger_velocities):
            line_vel.set_data(exp_time, self.Data['finger_kinematics'][time_slice,i+2])
        self.ax[3].set(xlabel='Time (sec)', ylabel='Flexion/Bin', title='Finger Velocities', xlim=time_lims,ylim=(-.125,.125)) # changed ylim a bit

        # clear any existing trial lines/texts
        for line in self.trial_lines:
            line.remove() 
        self.trial_lines.clear() 
        for txt in self.trial_texts:
            txt.remove()
        self.trial_texts.clear()

        # grab arrays of trial-start indices and counts
        trial_indexes = self.Data['trial_index']
        trial_numbers = self.Data['trial_number']
        
        # find all trial starts within the current window
        trial_indexes_in_range = trial_indexes[(trial_indexes > self.plot_start_index) & (trial_indexes < self.end_index)]
        
        # draw new trial start lines
        for index in trial_indexes_in_range:
            x_value = self.Data['time'][index] / 1000
            position = np.where(trial_indexes == index)[0][0]
            actual_trial = trial_numbers[position]

            for a in self.ax:
                line_sep = a.axvline(x=x_value, color='black', linewidth=2, alpha=0.3)
                self.trial_lines.append(line_sep)
            
            txt = self.ax[0].text(x_value, self.ax[0].get_ylim()[1]*0.95, str(actual_trial),
                                  color='black', fontsize=8, rotation=90, verticalalignment='top')
            self.trial_texts.append(txt)
        
        
        
        # print day and run in title
        day_str = self.filenames[self.fileidx][:10]

        run_ids = []
        if self.data_CO is not None and 'run_id' in self.data_CO:
            run_ids.append(str(self.data_CO['run_id']))
        if self.data_RD is not None and 'run_id' in self.data_RD:
            run_ids.append(str(self.data_RD['run_id']))

        run_label = "Runs" if len(run_ids) > 1 else "Run"
        run_str = ", ".join(run_ids) if run_ids else "unknown"
        self.fig.suptitle(f"Date: {day_str}     {run_label}: {run_str}", fontsize=14)
        
        with self.figoutput:
                clear_output(wait=True)
                display(self.fig)


    def shift_plot(self, change):
        # plot the requested trials
        self.plot_start_index += change
        self.plot_day()
        pass

    def shift_back(self, b):
        self.shift_plot(-500)
    
    def shift_forward(self, b):
        if self.end_index != len(self.Data['time']): # added this so it doesn't throw error messages at me
            self.shift_plot(500)
        else: 
            with self.text_output:
                print('Already done! Stop clicking the "→ (+500 bins)" button!') 

    def switch_TS(self, b):
        if self.both_TS_present:
            #switch targets styles
            self.current_TS = 'CO' if self.current_TS == 'RD' else 'RD'
            self.trial_num_printed = False
            with self.text_output:
                clear_output()
                print(f'Target Style = {self.current_TS}, {self.filenames[self.fileidx]}') # print current TS when TS switched
            self.plot_start_index = 0 # start at the beginning
            self.plot_day()
        else:
            with self.text_output:
                clear_output()
                print(f'Only {self.current_TS} available for {self.filenames[self.fileidx]}.')
    
    def save_day(self):
        status = self.good_bad.value
        note = self.notes.value if self.notes.value else ' '
        filename = self.filenames[self.fileidx]
        date = filename[0:10]

        if date not in self.results_df['Date'].values:
            print(pd.DataFrame([[date, status, note]], columns=self.results_df.columns))
            self.results_df = pd.concat([pd.DataFrame([[date, status, note]], columns=self.results_df.columns), self.results_df], ignore_index=True)
        else:
            self.results_df.loc[self.results_df['Date'] == date, ['Status','Note']] = [status, note]
            print(pd.DataFrame([[date, status, note]], columns=self.results_df.columns)) # had it print the new status-note

    def save_and_quit(self, b):
        clear_output()
        print('PLOTTING TERMINATED, SAVING AND QUITTING...')

        # save current day
        self.save_day()
        # TODO FINISH SAVING 
        # save current progress
        if not os.path.isfile(self.outputpath): # to avoid overwriting
            self.results_df.to_csv(self.outputpath, mode='w', header=True, index=True)
        else:
            self.results_df.to_csv(self.outputpath, mode='a', header=False, index=True)

        with open(self.outputpath, 'wb') as f:
            pickle.dump(self.fileidx, f)

        print('All done! Bye for now')
