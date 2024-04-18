import numpy as np
import matplotlib.pyplot as plt
import os

from datetime import datetime
from IPython.display import display, Audio
from scipy.io.wavfile import write
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

import banjo_string as bs
import membrane as mb

class Banjo:
    def __init__(self) -> None:
        self.sample_rate = 44100
        self.seconds = 3
        self.k = 1 / self.sample_rate
        self.samples = int(self.sample_rate * self.seconds)

        f_string = 293.7 # G 
        self.string = bs.BanjoString(self.k, f_string) #, 0.5, 0.005, 0.000415, 138.67, 3000000000, 1156)
        self.membrane = mb.Membrane(self.k) #, 20, 0.009, 0.000355, 0.4, 4000, 3800000000, 1380)
        
        self.calculate_connection_coefs()

    def calculate_connection_coefs(self):
        self.connection_s = self.k**2 / (self.string.rho * self.string.A * self.string.h)
        self.connection_m = self.k**2 / (self.membrane.rho * self.membrane.H * self.membrane.h)
        
        self.connection_idx_s = self.string.N - self.string.N // 5
        self.connection_idx_m = self.membrane.M // 2

    @staticmethod
    def fd_move(u,w):
        # Move the string grid one step forward in time
        u[:,2] = u[:,1]
        u[:,1] = u[:,0]

        # Move the membrane grid one step forward in time
        w[:,:,2] = w[:,:,1]
        w[:,:,1] = w[:,:,0]

        return u,w

    def generate_initial_states(self):
        u = np.zeros((self.string.N, 3))
        w = np.zeros((self.membrane.M, self.membrane.M, 3))

        u[3:10, 1] = np.hanning(7) * 0.6

        output = np.zeros(self.samples)
        return u, w, output

    def fd_connection(self, u, w, u_lc, w_lc):
        f = (w[w_lc, w_lc, 0] - u[u_lc, 0]) / ( self.connection_s + self.connection_m)

        u[u_lc, 0] += f * self.connection_s
        w[w_lc, w_lc, 0] -= f * self.connection_m
        return u, w

    def run(self):
        u, w, output = self.generate_initial_states()
        output_s = np.zeros(self.samples)
        output_m = np.zeros(self.samples)
        # wave_u = np.zeros((self.string.N, self.samples))
        # wave_w = np.zeros((self.membrane.M, self.membrane.M, self.samples))
    
        for i in range(self.samples):
            u = self.string.fd(u)
            w = self.membrane.fd(w)
            u, w = self.fd_connection(u, w, self.connection_idx_s, self.connection_idx_m)
            u, w = Banjo.fd_move(u,w)

            output[i] =  w[self.membrane.M // 2, self.membrane.M // 2, 0] + u[self.string.N // 2, 0] 
            output_s[i] = u[self.string.N // 2, 0]
            output_m[i] = w[self.membrane.M // 2, self.membrane.M // 2, 0]
            
            # wave_u[:,i] = u[:,0]
            # wave_w[:,:,i] = w[:,:,0]

        return output, output_s, output_m #, wave_u, wave_w


def plot_spectogram(output, sample_rate, title="Spectogram", path=None):
    plt.specgram(output, Fs=sample_rate)
    plt.title(title)

def plot_waveform(output, title="Waveform", path=None):
    plt.plot(output)
    plt.title(title)
    
def plot_fft(output, sample_rate, title="FFT", path=None):
    yf = np.fft.fft(output)
    plt.plot(np.log(np.abs(yf[:sample_rate//2])))
    plt.xscale("log")
    plt.title(title)

    
def plot_all(output, sample_rate, type, path=None):
    plt.subplot(3, 1, 1)
    plot_waveform(output, f"{type} Waveform", path)
    plt.subplot(3, 1, 2)
    plot_spectogram(output, sample_rate, f"{type} Spectogram", path)
    plt.subplot(3, 1, 3)
    plot_fft(output, sample_rate, f"{type} FFT", path)

    if path is not None:
        plt.savefig(path +  f"{type}" + '.png') 
        plt.clf()
    else: 
        plt.show()

def create_result_dir():
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S") # Format the current date and time as a string
    
    dir_path = "results/" +f"{timestamp}"
    os.makedirs(dir_path) # Create a directory with the timestamp in its name
    return dir_path

def animate_2d(wave):
    fig, ax = plt.subplots()
    wave = wave / np.max(np.abs(wave))
    img = ax.imshow(wave[:,:,0], interpolation='none', cmap='gray', animated=True, vmin=-1, vmax=1)
    ax.axis('off')
    
    def update(frame):
        img.set_array(wave[:,:,frame])
        return img,
    
    ani = FuncAnimation(fig, update, frames=wave.shape[2], blit=True)
    
    # Save the animation as a video file
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=7200)
    ani.save("wave_2d_animation.mp4", writer=writer)
    
from mpl_toolkits.mplot3d import Axes3D

def animate_3d(wave):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(-1, 1)

    x = np.arange(wave.shape[0])
    y = np.arange(wave.shape[1])
    x, y = np.meshgrid(x, y)
    z = wave[:,:,0]

    wave = wave / np.max(np.abs(wave))

    surf = ax.plot_surface(x, y, z, cmap='gray', vmin=-1, vmax=1)
    def update(frame):
        ax.clear()
        ax.set_zlim(-1, 1)

        z = wave[:,:,frame]
        surf = ax.plot_surface(x, y, z, cmap='gray', vmin=-1, vmax=1)
        
        return surf,

    ani = FuncAnimation(fig, update, frames=wave.shape[2], blit=True)

    # Save the animation as a video file
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=7200)
    ani.save("wave_3d_animation.mp4", writer=writer)


def animate_1d(wave):
    fig, ax = plt.subplots()
    line1, = ax.plot(wave[:, 0])
    ax.set_ylim(-1,1)
    
    def update(frame):
        line1.set_ydata(wave[:,frame])
        return line1,
    
    ani = FuncAnimation(fig, update, frames=wave.shape[1], blit=True)
    
    # Save the animation as a video file
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("wave_1d_animation.mp4", writer=writer)
    
# main function
if __name__ == "__main__":
    banjo = Banjo()
    output, output_s, output_m = banjo.run() # , wave_s, wave_m 
    
    path = create_result_dir() + "/"
    
    plot_all(output, banjo.sample_rate, "Banjo", path)
    plot_all(output_s, banjo.sample_rate, "String", path)
    plot_all(output_m, banjo.sample_rate, "Membrane", path)
    
    # animate_1d(wave_s)
    # animate_2d(wave_m)
    
    if path is not None:
        write(path + 'banjo.wav', banjo.sample_rate, output)
        write(path + 'string.wav', banjo.sample_rate, output_s)
        write(path + 'membrane.wav', banjo.sample_rate, output_m)

