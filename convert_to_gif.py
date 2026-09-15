# AV Overconfidence Death Zone Mapper
# Vamshikrishna Gadde | MS Robotics ASU
# Convert both demo videos to GitHub-native GIFs

import os
import imageio.v2 as imageio
from config import RESULTS_DIR

def convert(mp4_path, gif_path, fps=10, scale_width=800):
    print(f"Converting {mp4_path}")
    reader = imageio.get_reader(mp4_path)
    meta = reader.get_meta_data()
    src_fps = meta.get('fps', 24)
    step = max(1, round(src_fps / fps))

    frames = []
    for i, frame in enumerate(reader):
        if i % step != 0:
            continue
        h, w = frame.shape[:2]
        new_h = int(h * scale_width / w)
        frames.append(frame)

    print(f"  {len(frames)} frames at {fps}fps")
    imageio.mimsave(
        gif_path, frames, fps=fps,
        loop=0
    )
    size_mb = os.path.getsize(gif_path) / (1024 * 1024)
    print(f"  Saved: {gif_path} ({size_mb:.1f} MB)")


def main():
    videos_dir = os.path.join(RESULTS_DIR, "videos")
    gifs_dir = os.path.join(RESULTS_DIR, "gifs")
    os.makedirs(gifs_dir, exist_ok=True)

    convert(
        os.path.join(videos_dir, "av_overconfidence_demo.mp4"),
        os.path.join(gifs_dir, "av_overconfidence_demo.gif"),
        fps=8, scale_width=700
    )

    convert(
        os.path.join(videos_dir, "real_overconfidence_demo.mp4"),
        os.path.join(gifs_dir, "real_overconfidence_demo.gif"),
        fps=8, scale_width=700
    )


if __name__ == "__main__":
    main()