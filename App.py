import tkinter as tk
from tkinter import messagebox, ttk
from threading import Thread
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO

class MarioApp:
    def __init__(self, master):
        self.master = master
        master.title("An Intelligent Super Mario Bros. Game Agent")

        self.movement_var = tk.StringVar(value="right")  # Default movement type
        self.episodes_var = tk.IntVar(value=5)  # Default number of episodes
        self.create_selection_screen()

    def create_selection_screen(self):
        self.selection_label = tk.Label(self.master, text="Select Mario Agent Movement Type:")
        self.selection_label.pack(pady=10)

        movement_types = [("Right Only", "right"), ("Simple Movement", "simple"), ("Complex Movement", "complex"), ("Custom Movement", "custom")]
        for text, value in movement_types:
            rb = tk.Radiobutton(self.master, text=text, variable=self.movement_var, value=value)
            rb.pack()

        self.episodes_label = tk.Label(self.master, text="Enter No. of Episodes:")
        self.episodes_label.pack(pady=10)

        self.episodes_entry = tk.Entry(self.master, textvariable=self.episodes_var)
        self.episodes_entry.pack()

        self.start_button = tk.Button(self.master, text="Start Super Mario Bros. Game", command=self.start_game)
        self.start_button.pack(pady=25)

    def start_game(self):
        self.master.withdraw()  # Hide the selection screen
        self.play_game()

    def play_game(self):
        class CustomRewardAndDoneEnv(gym.Wrapper):
            def __init__(self, env=None):
                super(CustomRewardAndDoneEnv, self).__init__(env)
                self.current_score = 0
                self.current_x = 0
                self.current_x_count = 0
                self.max_x = 0
            def reset(self, **kwargs):
                self.current_score = 0
                self.current_x = 0
                self.current_x_count = 0
                self.max_x = 0
                return self.env.reset(**kwargs)
            def step(self, action):
                state, reward, done, info = self.env.step(action)
                reward += max(0, info['x_pos'] - self.max_x)
                if (info['x_pos'] - self.current_x) == 0:
                    self.current_x_count += 1
                else:
                    self.current_x_count = 0
                if info["flag_get"]:
                    reward += 500
                    done = True
                    print("GOAL")
                if info["life"] < 2:
                    reward -= 500
                    done = True
                self.current_score = info["score"]
                self.max_x = max(self.max_x, self.current_x)
                self.current_x = info["x_pos"]
                return state, reward / 10., done, info

        CUSTOM_MOVEMENT = [['left', 'A'], ['right', 'B'], ['right', 'A', 'B']]

        movement_type = self.movement_var.get()
        episodes = self.episodes_var.get()

        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        if movement_type == "right":
            env = JoypadSpace(env, RIGHT_ONLY)
            model_path = './models/model_right_only.zip'
        elif movement_type == "simple":
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            model_path = './models/model_simple_movement.zip'
        elif movement_type == "complex":
            env = JoypadSpace(env, COMPLEX_MOVEMENT)
            model_path = './models/model_complex_movement.zip'
        elif movement_type == "custom":
            env = JoypadSpace(env, CUSTOM_MOVEMENT)
            env = CustomRewardAndDoneEnv(env)
            model_path = './models/model_custom_movement.zip'
            

        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, shape=84)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order='last')

        model = PPO.load(model_path, env=env, clip_range=1)

        self.result_window = tk.Toplevel(self.master)
        self.result_window.title("Game Results")
        self.result_window.minsize(width=400, height=self.result_window.winfo_reqheight())
        
        for episode in range(1, episodes + 1):
            state = env.reset()
            done = False
            score = 0
            
            score_label = tk.Label(self.result_window, text=f'Score: {score}')
            score_label.pack()

            while not done:
                env.render()
                action, _ = model.predict(state)
                state, reward, done, info = env.step(action)
                score += reward
                score_label.config(text=f'Episode: {episode}    Score: {score}')
                self.result_window.update()
            
        env.close()
        self.master.deiconify()  # Restore the main window when the game is over
        self.start_button.config(state=tk.NORMAL)  # Enable the button after gameplay

def main():
    root = tk.Tk()
    root.minsize(width=500, height=root.winfo_reqheight())
    app = MarioApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()