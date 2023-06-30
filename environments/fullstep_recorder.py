from gym.wrappers.monitoring.video_recorder import VideoRecorder

from gym import logger
from typing import Optional

class FullStepRecorder(VideoRecorder):
    """Wrapper on the gym video recorder which saves all frames from a given step and supports selecting the camera id as a tuple of the return value from the render function.
    """

    def __init__(
        self,
        env,
        camera_id: int = 0,
        path: Optional[str] = None,
        metadata: Optional[dict] = None,
        enabled: bool = True,
        base_path: Optional[str] = None,
        **kwargs
    ):
        """Video recorder renders a nice movie of a rollout, frame by frame.

        Args:
            env (Env): Environment to take video of.
            camera_id (int): Camera ID to use for recording if a tuple of camera frames is returned. 0 indicates the first camera or tuple entry.
            path (Optional[str]): Path to the video file; will be randomly chosen if omitted.
            metadata (Optional[dict]): Contents to save to the metadata file.
            enabled (bool): Whether to actually record video, or just no-op (for convenience)
            base_path (Optional[str]): Alternatively, path to the video file without extension, which will be added.

        Raises:
            Error: You can pass at most one of `path` or `base_path`
            Error: Invalid path given that must have a particular file extension
        """
        temp = getattr(env, "render_mode", "rgb_array")
        env.render_mode = "rgb_array"
        super().__init__(env, path=path, metadata=metadata, enabled=enabled, base_path=base_path, **kwargs)
        env.render_mode = temp
        self.camera_id = camera_id

    def capture_frame(self, **kwargs):

        frames = self.env.render(**kwargs)
        if isinstance(frames, tuple):
            frames = frames[self.camera_id]

        if not self.functional:
            return
        if self._closed:
            logger.warn(
                "The video recorder has been closed and no frames will be captured anymore."
            )
            return
        logger.debug("Capturing video frame: path=%s", self.path)

        if frames is None:
            if self._async:
                return
            else:
                # Indicates a bug in the environment: don't want to raise
                # an error here.
                logger.warn(
                    "Env returned None on `render()`. Disabling further rendering for video recorder by marking as "
                    f"disabled: path={self.path} metadata_path={self.metadata_path}"
                )
                self.broken = True
        elif isinstance(frames, list):
            self.recorded_frames.extend(frames)
        else:
            self.recorded_frames.append(frames)