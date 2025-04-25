import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simplehand import SimpleHand
import matplotlib as mpl
from matplotlib.widgets import Button
mpl.rcParams['toolbar'] = 'None'
import random

from inputs.hand_tracker import HandTracker
from tasks.utils import HandTargetGenerator
from tasks.utils import visualize_neural_data
from tasks.utils import Clock

# Constants
SCREEN_WIDTH_IN = 12 #10
SCREEN_HEIGHT_IN = 8
NEURAL_SCREEN_WIDTH_IN = 10
NEURAL_SCREEN_HEIGHT_IN = 3
MAX_FPS = 30
DISP_FPS = False
DO_PLOT_NEURAL = False
NUM_CHANS_TO_PLOT = 20
NUM_NEURAL_HISTORY_PLOT = 100   # number of timepoints

CV2_CAMERA_ID = 0               # default camera id for cv2 (usually the webcam)


def hand_task(recorder, decoder, target_type="random", target_size = 0.15, hold_time = 500, target_dof = 1, is_demo = False, decoder_name = "GT"):
    print("\n\t✋  🤙 ✊️  Starting hand task, use ctrl-c to exit  ✌️ 👌 🖐  \n")
    
    # Target generation
    trial_timeout = 12000*target_dof # 10 sec per dof
    if target_type == "random":
        edge = 0.05  # prevent targets in the outer 5% of the screen
        target_gen = HandTargetGenerator(num_dof=target_dof, center_out=False, is_discrete=False, range=[edge, 1 - edge])

    elif target_type == "centerout":
        edge = 0.05  # prevent targets in the outer 5% of the screen
        target_gen = HandTargetGenerator(num_dof=target_dof, center_out=True, is_discrete=True, discrete_targs=None,range=[edge, 1 - edge])
    current_target = target_gen.generate_targets()

    
    # state vars
    recording = False
    online = False

    # init hand tracker
    hand_tracker = HandTracker(camera_id=CV2_CAMERA_ID, show_tracking=False)

    fig = plt.figure(figsize=(SCREEN_WIDTH_IN, SCREEN_HEIGHT_IN), num='Hand - Both')
    # gs = fig.add_gridspec(1, 2)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    ax_hand = fig.add_subplot(gs[0,0], projection='3d')
    ax_hand.set_title('Hand - DECODE')
    hand = SimpleHand(fig, ax_hand)
    hand.set_flex(0, 0, 0, 0, 0)
    hand.draw()
    
    ax_target = fig.add_subplot(gs[0,1], projection='3d')
    ax_target.set_title(f"Hand - TARGET: {current_target}")
    target_hand = SimpleHand(fig, ax_target)
    target_hand.set_flex(*current_target)
    target_hand.draw()


    # add button for recording
    if not is_demo:
        ax_record_button = fig.add_axes((0.05, 0.92, 0.15, 0.05))
        record_button = Button(ax_record_button, 'Start Recording', color="green")
    
    #useful text boxes
    decode_text = fig.text(0.25, 0.86, "Hand - DECODE", fontsize=12)
    results_text = fig.text(0.25, 0.95, f"Successes/Minute: ", fontsize=12)
    if is_demo:
        decoder_name_text = fig.text(0.45, 0.95, f"Using Decoder: ", fontsize=12)
        target_type_text = fig.text(0.7, 0.95, f"Target Type: ", fontsize=12)
        target_dof_text = fig.text(0.9, 0.95, f"DOF: ", fontsize=12)

    if not is_demo:
        def toggle_recording():
            nonlocal recording
            recording = not recording
            if recording:
                print("started recording")
                record_button.label.set_text("Stop Recording")
                record_button.color = "red"

            else:
                print("stopped recording")
                record_button.label.set_text("Start Recording")
                record_button.color = "green"
                recorder.save_to_file()

        record_button.on_clicked(lambda _: toggle_recording())

    # add button for online/offline
    if decoder is not None:
        ax_online_button = fig.add_axes((0.05, 0.92, 0.15, 0.05)) #fig.add_axes((0.25, 0.92, 0.15, 0.05))
        online_button = Button(ax_online_button, 'Go Online', color="green")

        def toggle_online():
            nonlocal online
            online = not online
            if online:
                online_button.label.set_text("Go Offline")
                online_button.color = "red"
                decoder.set_position(hand_tracker.get_hand_position())
            else:
                online_button.label.set_text("Go Online")
                online_button.color = "green"

        online_button.on_clicked(lambda _: toggle_online())

    # show the hand plot
    plt.show(block=False)

    # set up window for neural visualization
    fig_neural = None
    if decoder is not None:
        neural_history = collections.deque(maxlen=NUM_NEURAL_HISTORY_PLOT)
        if DO_PLOT_NEURAL:
            fig_neural, ax = plt.subplots(figsize=(NEURAL_SCREEN_WIDTH_IN, NEURAL_SCREEN_HEIGHT_IN),
                                        num='Neural Data Visualization (first 20 channels)')
            ani = FuncAnimation(fig_neural,
                                lambda i: visualize_neural_data(ax, neural_history, NUM_CHANS_TO_PLOT),
                                interval=1000 / MAX_FPS,
                                cache_frame_data=False)
            plt.show(block=False)  # non-blocking, continues with script execution

    # main loop
    clock = Clock(disp_fps=DISP_FPS)
    trial_start_time = 0
    
    #demo metrics
    total_trials = 11
    total_successful = 0
    trial_idx = 0
    first_success_time = 0
    trial_times = np.zeros(total_trials)
    if is_demo:
        if decoder is not None:
            toggle_online()
    while trial_idx < total_trials: #since trial idx is only updated in demo, if not demo then effectively while true

        # get hand position
        hand_pos_true = hand_tracker.get_hand_position()

        if online:
            # run the decoder to get cursor position
            hand_pos_in = np.array(hand_pos_true)
            hand_pos = decoder.decode(hand_pos_in)
            neural_history.append(decoder.get_recent_neural())

        else:
            # offline - just use the true hand position
            hand_pos = hand_pos_true  
                
        # draw hand
        azim, elev = ax_hand.azim, ax_hand.elev     # get current view
        ax_hand.clear()
        hand.set_flex(*hand_pos)
        hand.draw()
        ax_hand.view_init(elev, azim)               # set view back to what it was
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        decode_text.set_text(f"Hand - DECODE: {np.round(hand_pos, 2)}")
        results_text.set_text(f"Successes/Minute: {np.round(60000*total_successful/(clock.get_time_ms()-first_success_time), 1)}" ) #starts after first success
        if is_demo:
            decoder_name_text.set_text(f"Using Decoder: {decoder_name}" )
            target_type_text.set_text(f"Target Type: {target_type}" )
            target_dof_text.set_text(f"DOF: {target_dof}" )
        fig.canvas.draw_idle()
        
        if max(abs(np.subtract(hand_pos, current_target))) < target_size:
            if time_entered_target == 0:
                time_entered_target = clock.get_time_ms()
            elif clock.get_time_ms() - time_entered_target >= hold_time:
                if total_successful == 0:
                    first_success_time = clock.get_time_ms()
                total_successful +=1
                if is_demo:
                    trial_times[trial_idx] = clock.get_time_ms() - trial_start_time
                    trial_idx += 1
                current_target = target_gen.generate_targets()
                trial_start_time = clock.get_time_ms()
                azim, elev = ax_target.azim, ax_target.elev     # get current view
                ax_target.clear()
                target_hand.set_flex(*current_target)
                target_hand.draw()
                ax_target.view_init(elev, azim)               # set view back to what it was
                fig.canvas.draw()
                fig.canvas.flush_events()
                ax_target.set_title(f"Hand - TARGET: {current_target}")
        else: 
            time_entered_target = 0
        
        # trial timeout
        if clock.get_time_ms() - trial_start_time >= trial_timeout:
            current_target = target_gen.generate_targets()
            if is_demo:
                trial_times[trial_idx] = trial_timeout
                trial_idx += 1
            trial_start_time = clock.get_time_ms()
            azim, elev = ax_target.azim, ax_target.elev     # get current view
            ax_target.clear()
            target_hand.set_flex(*current_target)
            target_hand.draw()
            ax_target.view_init(elev, azim)               # set view back to what it was
            fig.canvas.draw()
            fig.canvas.flush_events()
            ax_target.set_title(f"Hand - TARGET: {current_target}")
            
        
            

        # draw neural data
        if DO_PLOT_NEURAL and fig_neural is not None:
            fig_neural.canvas.draw()
            fig_neural.canvas.flush_events()

        # record data if recording is active
        if not is_demo:
            if recording:
                recorder.record(clock.get_time_ms(),
                                int(clock.get_time_ms() / 1000) + 1,    # dummy trials, once per second
                                hand_pos,
                                [0, 0, 0, 0, 0],                        # dummy target position
                                online)

        # update clock to limit frame rate (usually we're well below this)
        clock.tick(MAX_FPS)
    plt.close(fig)
    return trial_times
