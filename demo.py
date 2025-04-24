import argparse
import pickle
from data_recorder import DataRecorder
from inputs.decoder import RealTimeDecoder
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def load_decoder(decoder_name, num_dof, integration_beta):
    # load in a pre-trained decoder
    if not decoder_name.endswith(".pkl"):
        decoder_name += ".pkl"
    with open(f"data/trained_decoders/{decoder_name}", "rb") as f:
        model, neuralsim, neural_scaler, output_scaler, seq_len = pickle.load(f)
    decoder = RealTimeDecoder(num_dof, model, neuralsim, neural_scaler, output_scaler, seq_len, integration_beta)
    print(f"Loaded decoder: {decoder_name}")
    return decoder


def get_task(task_choice):
    if task_choice == "cursor":
        from tasks import cursor2d
        num_dof = 2
        return cursor2d.cursor_task, num_dof

    elif task_choice == "hand":
        from tasks import handtask
        num_dof = 5
        return handtask.hand_task, num_dof

    else:
        raise ValueError(f"Invalid task choice: {task_choice}")
    
def show_popup(message, duration=5):
    remaining_time = [duration]  # Mutable to allow updating

    # Create figure
    plt.ion()
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Welcome to the BCI Simulator!", fontsize=18)
    fig.subplots_adjust(bottom=0.2)

    # Main message in center
    ax_main = fig.add_axes([0, 0.3, 1, 0.5])
    ax_main.axis('off')
    msg_text = ax_main.text(0.5, 0.5, message, fontsize=20, ha='center', va='center')

    # Timer display at bottom
    ax_timer = fig.add_axes([0, 0.15, 1, 0.1])
    ax_timer.axis('off')
    timer_text = ax_timer.text(0.5, 0.5, "", fontsize=14, ha='center', va='center')

    # Button below the timer
    ax_button = fig.add_axes([0.4, 0.02, 0.2, 0.1])
    button = Button(ax_button, 'Extend 30s')

    def extend_time(event):
        remaining_time[0] += 30
        print("Extended time by 30 seconds.")

    button.on_clicked(extend_time)

    # Countdown loop
    while remaining_time[0] > 0:
        timer_text.set_text(f"Closing in {remaining_time[0]} second(s)...")
        plt.draw()
        plt.pause(1)
        remaining_time[0] -= 1

    plt.close()
    plt.ioff()


def main():
    parser = argparse.ArgumentParser(description="BCI Simulator")
    parser.add_argument("-t", "--task", default="hand", choices=["cursor", "hand"],
                        help="Task choice: cursor or hand.")
    parser.add_argument("-d", "--decoder", default=None,
                        help="Name of the decoder file (e.g., rnndecoder1). If specified, a real-time wrapper will load the decoder.")
    parser.add_argument("-tt", "--target_type", default="random", choices=["random", "centerout"],
                        help="Target type: random or centerout.")
    parser.add_argument("-tdof", "--target_dof", default=1, type=int, choices=[1,2,3],
                        help="Target dof: 1, 2, 3.")
    parser.add_argument("-thold", "--target_hold_time", type=float, default=500,
                        help="Target hold time in milliseconds")
    parser.add_argument("-tsize", "--target_size", type=float, default=0.15,
                        help="Target size, range from 0.05 to 0.25")
    # add argument for the decoder integration beta
    parser.add_argument("-b", "--integration_beta", default=0.98, type=float,
                        help="Integration beta: the percentage of decoded position that is integrated velocity.")
    args = parser.parse_args()

    # get task
    task, num_dof = get_task(args.task)
    
    #DEMO: cycle through all decoders and target styles and compare to GT performance
    decoders = ['GT','handrnn', 'handlstm', 'handgru']
    target_types = ["random", "centerout"]
    show_popup("Instructions...", duration=5)
    for target_dof in [1,2,3]:
        for target_type in target_types:
            for decoder_name in decoders:
                decoder = None
                if decoder_name != 'GT':
                    decoder = load_decoder(decoder_name, num_dof, args.integration_beta)
    
                trial_times = task(DataRecorder(), decoder, target_type=args.target_type, target_size = args.target_size, hold_time = args.target_hold_time, target_dof = args.target_dof,is_demo = True, decoder_name = decoder_name)
                print(trial_times)
                ## todo 
                # remove the first trial, force online for decoders

if __name__ == "__main__":
    main()
