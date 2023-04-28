import av

def av_decode_video(video_path):
    try:
        with av.open(video_path) as container:
            frames = []
            for frame in container.decode(video=0):
                frames.append(frame.to_rgb().to_ndarray())
        return frames
    except Exception:
        assert len(frame) != 0
        return frames