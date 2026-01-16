from __future__ import annotations

from pathlib import Path

from labelformat.formats.youtubevis import YouTubeVISInstanceSegmentationTrackInput

# Update this path to your YouTube-VIS JSON file.
INPUT_PATH = Path("/Users/jonaswurst/Lightly/dataset_examples/youtube_vis_50_videos/train/instances_50.json")


def main() -> None:
    label_input = YouTubeVISInstanceSegmentationTrackInput(input_file=INPUT_PATH)
    videos = list(label_input.get_videos())
    labels = list(label_input.get_labels())
    print(f"videos={len(videos)} labels={len(labels)}")
    if labels:
        first = labels[0]
        print(f"first video id={first.video.id} objects={len(first.objects)}")
        if first.objects:
            segs = first.objects[0].segmentations
            print(f"first track frames={len(segs)}")


if __name__ == "__main__":
    main()
