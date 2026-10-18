"""Contact sheet of N evenly spaced frames: frames.py <video.mp4> <out.png> [n]"""
import sys
import imageio.v3 as iio
import numpy as np
frames = iio.imread(sys.argv[1])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
pick = frames[np.linspace(0, len(frames) - 1, n).astype(int)]
pick = pick[:, ::2, ::2]
rows = [np.concatenate(pick[i:i + 4], 1) for i in range(0, n, 4)]
iio.imwrite(sys.argv[2], np.concatenate(rows, 0))
