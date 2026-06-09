"""Export visualization to images and videos."""

import os


def save_screenshot(plotter, path):
    """Save current view as an image.

    Args:
        plotter: pyvista.Plotter instance.
        path: output file path (e.g. screenshot.png).
    """
    plotter.screenshot(path)


def save_animation(plotter, path, n_frames=36, view="orbit"):
    """Rotate the view and save as an animation.

    Args:
        plotter: pyvista.Plotter instance.
        path: output file path (e.g. animation.mp4).
        n_frames: number of frames to render.
        view: rotation style ("orbit" supported).
    """
    plotter.open_movie(path)
    if view == "orbit":
        plotter.orbit_on_path(n_frames=n_frames, viewup=[0, 0, 1])
    plotter.close()


def video_to_images(video_path, output_dir):
    """Extract frames from a video file.

    Args:
        video_path: path to input video file.
        output_dir: directory to write frame images (named 0.jpg, 1.jpg, …).
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"{i}.jpg"), frame)
        i += 1
    cap.release()


def images_to_video(image_dir, output_path, fps=30):
    """Create a video from a sequence of images.

    Args:
        image_dir: directory containing frame images.
        output_path: path for the output video file.
        fps: frames per second.
    """
    import cv2

    image_list = sorted(
        [os.path.join(image_dir, f) for f in os.listdir(image_dir)],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
    )
    if not image_list:
        raise ValueError(f"No images found in {image_dir}")

    img = cv2.imread(image_list[0])
    h, w = img.shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for path in image_list:
        frame = cv2.imread(path)
        out.write(frame)
    out.release()
