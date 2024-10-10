import cv2
from matplotlib import animation
from IPython.display import HTML
from tempfile import NamedTemporaryFile
import base64
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns
from tqdm.auto import trange

import torch
from torch.distributions import MultivariateNormal, Normal

# Specify that we want our tensors on the GPU and in float32
device = torch.device('cpu')
dtype = torch.float32

# Helper function to convert between numpy arrays and tensors
to_t = lambda array: torch.tensor(array, device=device, dtype=dtype)
from_t = lambda tensor: tensor.to("cpu").detach().numpy().astype(np.float64)


sns.set_context("notebook")

# initialize a color palette for plotting
palette = sns.xkcd_palette(["windows blue",
                            "red",
                            "medium green",
                            "dusty purple",
                            "greyish",
                            "orange",
                            "amber",
                            "clay",
                            "pink"])


def combine(Ta, a, Tb, b):
    assert a or b
    if a is None:
        return Tb, b
    elif b is None:
        return Ta, a
    else:
        return tuple((Ta * ai + Tb * bi) / (Ta + Tb) for ai, bi in zip(a, b))


_VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def _anim_to_html(anim, fps=20):
    # todo: todocument
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=fps)
            video = open(f.name, "rb").read()
        anim._encoded_video = base64.b64encode(video)

    return _VIDEO_TAG.format(anim._encoded_video.decode('ascii'))

def _display_animation(anim, fps=30, start=0, stop=None):
    plt.close(anim._fig)
    return HTML(_anim_to_html(anim, fps=fps))

def play(movie, fps=30, speedup=1, fig_height=6,
         filename=None, show_time=False, show=True):
    # First set up the figure, the axis, and the plot element we want to animate
    T, Py, Px = movie.shape[:3]
    fig, ax = plt.subplots(1, 1, figsize=(fig_height * Px/Py, fig_height))
    im = plt.imshow(movie[0], interpolation='None', cmap=plt.cm.gray)

    if show_time:
        tx = plt.text(0.75, 0.05, 't={:.3f}s'.format(0),
                    color='white',
                    fontdict=dict(size=12),
                    horizontalalignment='left',
                    verticalalignment='center',
                    transform=ax.transAxes)
    plt.axis('off')

    def animate(i):
        im.set_data(movie[i * speedup])
        if show_time:
            tx.set_text("t={:.3f}s".format(i * speedup / fps))
        return im,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate,
                                   frames=T // speedup,
                                   interval=1,
                                   blit=True)
    plt.close(anim._fig)

    # save to mp4 if filename specified
    if filename is not None:
        with open(filename, "wb") as f:
            anim.save(f.name, fps=fps, extra_args=['-vcodec', 'libx264'])

    # return an HTML video snippet
    if show:
        print("Preparing animation. This may take a minute...")
        return HTML(_anim_to_html(anim, fps=30))

def initialize_posteriors(dataset, num_states, seed=0):
    torch.manual_seed(seed)
    posteriors = []
    for data in dataset:
        expected_states = torch.rand(len(data["data"]), num_states)
        expected_states /= expected_states.sum(axis=1, keepdims=True)
        posteriors.append(dict(posterior_prob=expected_states,
                               marginal_ll=-torch.inf))
    return posteriors
    
def plot_data_and_states(data, states,
                         spc=4, slc=slice(0, 900),
                         title=None):
    times = data["times"][slc]
    labels = data["labels"][slc]
    x = data["data"][slc]
    num_timesteps, data_dim = x.shape

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(states[None, slc],
              cmap="cubehelix", aspect="auto",
              extent=(0, times[-1] - times[0], -data_dim * spc, spc))

    ax.plot(times - times[0],
            x - spc * np.arange(data_dim),
            ls='-', lw=3, color='w')
    ax.plot(times - times[0],
            x - spc * np.arange(data_dim),
            ls='-', lw=2, color=palette[0])

    ax.set_yticks(-spc * np.arange(data_dim))
    ax.set_yticklabels(np.arange(data_dim))
    ax.set_ylabel("principal component")
    ax.set_xlim(0, times[-1] - times[0])
    ax.set_xlabel("time [ms]")

    if title is None:
        ax.set_title("data and discrete states")
    else:
        ax.set_title(title)


def extract_syllable_slices(state_idx,
                            posteriors,
                            pad=30,
                            num_instances=50,
                            min_duration=5,
                            max_duration=45,
                            seed=0):
    # Find all the start indices and durations of specified state
    state_idx = from_t(state_idx)
    all_mouse_inds = []
    all_starts = []
    all_durations = []
    for mouse, posterior in enumerate(posteriors):
        expected_states = from_t(posterior["posterior_prob"])
        states = np.argmax(expected_states, axis=1)
        states = np.concatenate([[-1], states, [-1]])
        starts = np.where((states[1:] == state_idx) \
                          & (states[:-1] != state_idx))[0]
        stops = np.where((states[:-1] == state_idx) \
                         & (states[1:] != state_idx))[0]
        durations = stops - starts
        assert np.all(durations >= 1)
        all_mouse_inds.append(mouse * np.ones(len(starts), dtype=int))
        all_starts.append(starts)
        all_durations.append(durations)

    all_mouse_inds = np.concatenate(all_mouse_inds)
    all_starts = np.concatenate(all_starts)
    all_durations = np.concatenate(all_durations)

    # Throw away ones that are too short or too close to start.
    # TODO: also throw away ones close to the end
    valid = (all_durations >= min_duration) \
            & (all_durations < max_duration) \
            & (all_starts > pad)

    num_valid = np.sum(valid)
    all_mouse_inds = all_mouse_inds[valid]
    all_starts = all_starts[valid]
    all_durations = all_durations[valid]

    # Choose a random subset to show
    rng = npr.RandomState(seed)
    subset = rng.choice(num_valid,
                        size=min(num_valid, num_instances),
                        replace=False)

    all_mouse_inds = all_mouse_inds[subset]
    all_starts = all_starts[subset]
    all_durations = all_durations[subset]

    # Extract slices for each mouse
    slices = []
    for mouse in range(len(posteriors)):
        is_mouse = (all_mouse_inds == mouse)
        slices.append([slice(start, start + dur) for start, dur in
                       zip(all_starts[is_mouse], all_durations[is_mouse])])

    return slices


def make_crowd_movie(state_idx,
                     dataset,
                     posteriors,
                     pad=30,
                     raw_size=(512, 424),
                     crop_size=(80, 80),
                     offset=(50, 50),
                     scale=.5,
                     min_height=10,
                     **kwargs):
    '''
    Adapted from https://github.com/dattalab/moseq2-viz/blob/release/moseq2_viz/viz.py

    Creates crowd movie video numpy array.
    Parameters
    ----------
    dataset (list of dicts): list of dictionaries containing data
    slices (np.ndarray): video slices of specific syllable label
    pad (int): number of frame padding in video
    raw_size (tuple): video dimensions.
    frame_path (str): variable to access frames in h5 file
    crop_size (tuple): mouse crop size
    offset (tuple): centroid offsets from cropped videos
    scale (int): mouse size scaling factor.
    min_height (int): minimum max height from floor to use.
    kwargs (dict): extra keyword arguments
    Returns
    -------
    crowd_movie (np.ndarray): crowd movie for a specific syllable.
    '''
    slices = extract_syllable_slices(state_idx, posteriors)

    xc0, yc0 = crop_size[1] // 2, crop_size[0] // 2
    xc = np.arange(-xc0, xc0 + 1, dtype='int16')
    yc = np.arange(-yc0, yc0 + 1, dtype='int16')

    durs = []
    for these_slices in slices:
        for slc in these_slices:
            durs.append(slc.stop - slc.start)

    if len(durs) == 0:
        print("no valid syllables found for state", state_idx)
        return
    max_dur = np.max(durs)

    # Initialize the crowd movie
    crowd_movie = np.zeros((max_dur + pad * 2, raw_size[1], raw_size[0], 3),
                            dtype='uint8')

    for these_slices, data in zip(slices, dataset):
        for slc in these_slices:
            lpad = min(pad, slc.start)
            rpad = min(pad, len(data['frames']) - slc.stop)
            dur = slc.stop - slc.start
            padded_slc = slice(slc.start - lpad, slc.stop + rpad)
            centroid_x = from_t(data['centroid_x_px'][padded_slc] + offset[0])
            centroid_y = from_t(data['centroid_y_px'][padded_slc] + offset[1])
            angles = np.rad2deg(from_t(data['angles'][padded_slc]))
            frames = data['frames'].detach().numpy()
            frames = (frames[padded_slc] / scale).astype('uint8')
            flips = np.zeros(angles.shape, dtype='bool')

            for i in range(lpad + dur + rpad):
                if np.any(np.isnan([centroid_x[i], centroid_y[i]])):
                    continue

                rr = (yc + centroid_y[i]).astype('int16')
                cc = (xc + centroid_x[i]).astype('int16')

                if (np.any(rr < 1)
                    or np.any(cc < 1)
                    or np.any(rr >= raw_size[1])
                    or np.any(cc >= raw_size[0])
                    or (rr[-1] - rr[0] != crop_size[0])
                    or (cc[-1] - cc[0] != crop_size[1])):
                    continue

                # rotate and clip the current frame
                new_frame_clip = frames[i][:, :, None] * np.ones((1, 1, 3))
                rot_mat = cv2.getRotationMatrix2D((xc0, yc0), angles[i], 1)
                new_frame_clip = cv2.warpAffine(new_frame_clip.astype('float32'),
                                                rot_mat, crop_size).astype(frames.dtype)

                # overlay a circle on the mouse
                if i >= lpad and i <= pad + dur:
                    cv2.circle(new_frame_clip, (xc0, yc0), 3,
                               (255, 0, 0), -1)

                # superimpose the clipped mouse
                old_frame = crowd_movie[i]
                new_frame = np.zeros_like(old_frame)
                new_frame[rr[0]:rr[-1], cc[0]:cc[-1]] = new_frame_clip

                # zero out based on min_height before taking the non-zeros
                new_frame[new_frame < min_height] = 0
                old_frame[old_frame < min_height] = 0

                new_frame_nz = new_frame > 0
                old_frame_nz = old_frame > 0

                blend_coords = np.logical_and(new_frame_nz, old_frame_nz)
                overwrite_coords = np.logical_and(new_frame_nz, ~old_frame_nz)

                old_frame[blend_coords] = .5 * old_frame[blend_coords] \
                    + .5 * new_frame[blend_coords]
                old_frame[overwrite_coords] = new_frame[overwrite_coords]

                crowd_movie[i] = old_frame

    return crowd_movie

def nearest_positive_definite(A):
    """
    Find the nearest positive-definite matrix to input A.
    """
    B = (A + A.T) / 2
    _, s, V = torch.svd(B)
    H = V @ torch.diag(s) @ V.T
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if is_positive_definite(A3):
        return A3
    spacing = torch.eye(A.shape[0]) * 1e-4
    k = 1
    while not is_positive_definite(A3):
        min_eig = torch.min(torch.linalg.eigvalsh(A3))
        A3 += spacing * (-min_eig * k**2)
        k += 1
    return A3

def is_positive_definite(B):
    try:
        _ = torch.linalg.cholesky(B)
        return True
    except RuntimeError:
        return False

def sticky_transitions(num_states, stickiness=0.95):
    P = stickiness * torch.eye(num_states)
    P += (1 - stickiness) / (num_states - 1) * (1 - torch.eye(num_states))
    return P

def random_args(num_timesteps, num_states, seed=0,
                offset=0, scale=1):
    torch.manual_seed(seed)
    pi = torch.ones(num_states) / num_states
    P = sticky_transitions(num_states)
    log_likes = offset + scale * Normal(0,1).sample((num_timesteps, num_states))
    return pi, P, log_likes

def plot_average_pcs(state_idx,
                     dataset,
                     posteriors,
                     spc=4,
                     pad=30,
                     fps=30):
    '''
    '''
    # Find slices for this state
    slices = extract_syllable_slices(state_idx, posteriors, num_instances=1000)
    data_dim = dataset[0]['data'].shape[1]

    # Find maximum duration
    durs = []
    num_slices = 0
    for these_slices in slices:
        for slc in these_slices:
            durs.append(slc.stop - slc.start)
            num_slices += 1
    if num_slices == 0:
        print("no valid syllables found for state", state_idx)
        return
    max_dur = np.max(durs)

    # Initialize timestamps
    times = np.arange(-pad, max_dur + pad) / fps
    exs = np.nan * np.ones((num_slices, 2 * pad + max_dur, data_dim))
    counter = 0

    # Make figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for these_slices, data, posterior in zip(slices, dataset, posteriors):
        for slc in these_slices:
            lpad = min(pad, slc.start)
            rpad = min(pad, len(data['data']) - slc.stop)
            dur = slc.stop - slc.start
            padded_slc = slice(slc.start - lpad, slc.stop + rpad)

            x = data['data'][padded_slc]
            exs[counter][(pad - lpad):(pad - lpad + len(x))] = x
            counter += 1

            # Plot single example
            # ax.plot(times[(pad - lpad):(pad - lpad + len(x))],
            #         x - spc * np.arange(data_dim),
            #         ls='-', lw=.5, color='k')

    # take the mean and standard deviation
    ex_mean = np.nanmean(exs, axis=0)
    ex_std = np.nanstd(exs, axis=0)

    for d in range(data_dim):
        ax.fill_between(times,
                        ex_mean[:, d] - 2 * ex_std[:, d] - spc * d,
                        ex_mean[:, d] + 2 * ex_std[:, d] - spc * d,
                        color='k', alpha=0.25)

        ax.plot(times, ex_mean[:, d] - spc * d, '-k', lw=2)

    ax.plot([0, 0], [-spc * data_dim, spc], '-r', lw=2)
    ax.set_yticks(-spc * np.arange(data_dim))
    ax.set_yticklabels(np.arange(data_dim))
    ax.set_ylim(-spc * data_dim, spc)
    ax.set_ylabel("principal component")
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("$\Delta t$ [ms]")

    ax.set_title("Average PCs for State {}".format(state_idx))