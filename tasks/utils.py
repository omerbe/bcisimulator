import numpy as np
import time


class Clock:
    # Simple clock class to maintain a constant (max) frame rate
    def __init__(self, disp_fps=False):
        self.start_time = time.time_ns() // 1_000_000
        self.last_time = time.time_ns() // 1_000_000
        self.disp_fps = disp_fps

    def tick(self, fps):
        # sleep to maintain fps
        now = time.time_ns() // 1_000_000
        if self.disp_fps:
            print(f'fps: {1000 / (now - self.last_time):5.1f}', end='\r')
        time.sleep(max(0, 1000 / fps - (now - self.last_time)))
        self.last_time = now

    def get_time_ms(self):
        return time.time_ns() // 1_000_000 - self.start_time


class TargetGenerator:
    # Generates target positions for the task, of random or discrete positions

    def __init__(self, num_dof=1, center_out=False, is_discrete=False, discrete_targs=None, continuous_range=None):
        """
        :param num_dof (int):           Number of degrees of freedom (i.e. how many targets to make)
        :param center_out (bool):       If True, alternates between the center position (0.5) and the other targets
        :param is_discrete (bool):      If True, will choose targets from the discrete_targs list. If False, will
                                        randomly choose targets from the continuous_range.
        :param discrete_targs (list):   List of target positions to choose from in discrete mode. Each target should be
                                        a tuple of length num_dof.
        :param continuous_range (list): List with the upper and lower limits for continuous targets. Defaults to [0, 1]
        """
        self.num_dof = num_dof
        self.center_out = center_out
        self.is_discrete = is_discrete
        self.discrete_targs = np.array(discrete_targs)
        self.cont_range = continuous_range if continuous_range else [0, 1]

        self.at_center = False
        self.target_pos = None

    def reset(self):
        self.at_center = False
        self.target_pos = None

    def generate_targets(self):
        if self.center_out:
            if not self.at_center:
                self.target_pos = 0.5 * np.ones(self.num_dof)
                self.at_center = True
            else:
                # self.target_pos = np.array(np.random.choice(self.discrete_targs))
                self.target_pos = self.discrete_targs[np.random.choice(self.discrete_targs.shape[0])]
                self.at_center = False

        else:
            self.target_pos = np.array([np.random.uniform(self.cont_range[0], self.cont_range[1])
                                        for _ in range(self.num_dof)])
        return self.target_pos
    
    
    
class HandTargetGenerator:
    # Generates target positions for the task, of random or discrete positions

    def __init__(self, num_dof=1, center_out=False, is_discrete=False, discrete_targs=None, range=None):
        """
        :param num_dof (int):           Number of degrees of freedom (Index, MRS, Thumb)
        :param center_out (bool):       If True, alternates between the center position (0.5) and the other targets
        :param is_discrete (bool):      If True, will choose targets from the discrete_targs list. If False, will
                                        randomly choose targets from the continuous_range.
        :param discrete_targs (list):   List of target positions to choose from in discrete mode. Each target should be
                                        a tuple of length num_dof.
        :param continuous_range (list): List with the upper and lower limits for continuous targets. Defaults to [0, 1]
        """
        self.num_dof = num_dof
        self.center_out = center_out
        self.is_discrete = is_discrete
        self.range = range if range else [0, 1]
        self.discrete_targs = self.setup_discrete_targets(num_targets = 10)

        self.at_center = False
        self.target_pos = None

    def reset(self):
        self.at_center = False
        self.target_pos = None
        
    def setup_discrete_targets(self, num_targets, remove_center=False):
        """ function to automatically calculate equally spaced targets """
        targets = np.array(np.linspace(self.range[0], self.range[1], num_targets))
        if remove_center:
            targets = np.array([target for target in targets if (target != 0.5)])
        return targets
        
    # convert num DOF to 5 finger targets
    def adjust_targets(self):
        if self.num_dof == 1:
            idx = [0]*5
        elif self.num_dof == 2:
            idx = [0]*1+[1]*4
        elif self.num_dof == 3:
            idx = [0]*1+[1]*1+[2]*3
        else:
            raise ValueError(f"Invalid target dof: {self.num_dof}")
        self.target_pos = [self.target_pos[i] for i in idx]
        
        self.target_pos[0] = max(0.3, self.target_pos[0]) # set thumb floor at 0.3
        self.target_pos[4] = max(.1, self.target_pos[4]) # set pinky floor at 0.1 
        
        self.target_pos = list(np.round(self.target_pos, 2))

    def generate_targets(self):
        if self.center_out:
            if not self.at_center:
                self.target_pos = 0.5 * np.ones(self.num_dof)
                self.at_center = True
            else:
                self.target_pos = np.random.choice(self.discrete_targs, self.num_dof)
                
                self.at_center = False

        else:
            self.target_pos = np.array([np.random.uniform(self.range[0], self.range[1])
                                        for _ in range(self.num_dof)])
        self.adjust_targets()
        return self.target_pos
    


def visualize_neural_data(ax, neural_history, num_chans_to_plot=20):
    ax.clear()  # clear previous data
    if neural_history:
        data = np.array(neural_history)
        ypos = 0
        for ch in range(min(data.shape[1], num_chans_to_plot)):
            ax.plot(data[:, ch] + ypos)
            ypos += 3
    ax.set_position([0, 0, 1, 1])
    ax.axis('off')
