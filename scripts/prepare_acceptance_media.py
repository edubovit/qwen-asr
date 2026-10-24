import json
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    fixtures = root / ".local" / "fixtures"
    output = root / ".local" / "verification" / "media"
    output.mkdir(parents=True, exist_ok=True)

    def ffmpeg(*args):
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args],
            check=True,
        )

    ffmpeg("-i", str(fixtures / "en_fleurs.wav"), "-c:a", "libmp3lame",
           str(output / "English recording.mp3"))
    ffmpeg("-i", str(fixtures / "ru_fleurs.wav"), "-c:a", "pcm_s16le",
           str(output / "Русская запись.wav"))
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=160x90:r=1",
           "-i", str(fixtures / "en_fleurs.wav"), "-c:v", "libx264", "-preset", "ultrafast",
           "-c:a", "aac", "-shortest", str(output / "English video.mp4"))
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=160x90:r=10:d=18",
           "-itsoffset", "2", "-i", str(fixtures / "en_fleurs.wav"),
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "mpeg4", "-c:a", "pcm_s16le",
           str(output / "Delayed English video.mkv"))
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=start_time:stream=codec_type,start_time", "-of", "json",
         str(output / "Delayed English video.mkv")],
        check=True, capture_output=True, text=True,
    )
    timeline = json.loads(probe.stdout)
    assert float(timeline["format"]["start_time"]) == 0
    assert next(float(s["start_time"]) for s in timeline["streams"]
                if s["codec_type"] == "audio") == 2
    ffmpeg("-stream_loop", "3", "-i", str(fixtures / "mixed_ru_en.wav"),
           "-c:a", "pcm_s16le", str(output / "mixed repeated.wav"))
    ffmpeg("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "7200",
           "-c:a", "pcm_s16le", str(output / "two hours silence.wav"))
    (output / "corrupt.wav").write_bytes(b"Not an audio file")
    (output / "empty.wav").write_bytes(b"")
    print(output)


if __name__ == "__main__":
    main()
