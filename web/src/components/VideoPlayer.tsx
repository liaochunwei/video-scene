import { useRef, useEffect, useState, useCallback } from 'react';
import type { SearchResult, MoreSegment } from '../types';
import { Badge } from './ui/badge';

interface VideoPlayerProps {
    result: SearchResult | null;
    paused?: boolean;
}

interface SegmentThumb {
    start_time: number;
    end_time: number;
    keyframe_url: string;
}

function formatTime(seconds: number): string {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

export function VideoPlayer({ result, paused }: VideoPlayerProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const [activeIndex, setActiveIndex] = useState(0);

    const thumbnails: SegmentThumb[] = result
        ? [
              {
                  start_time: result.start_time ?? 0,
                  end_time: result.end_time ?? 0,
                  keyframe_url: result.keyframe_url,
              },
              ...result.more
                  .filter((m) => m.start_time != null && m.end_time != null)
                  .map((m: MoreSegment) => ({
                      start_time: m.start_time!,
                      end_time: m.end_time!,
                      keyframe_url: m.keyframe_url,
                  })),
          ]
        : [];

    // Reset active segment when result changes
    useEffect(() => {
        setActiveIndex(0);
    }, [result]);

    // Pause video when paused prop becomes true
    useEffect(() => {
        const video = videoRef.current;
        if (video && paused) {
            video.pause();
        }
    }, [paused]);

    useEffect(() => {
        const video = videoRef.current;
        if (!video || !result) return;

        video.src = `/api/video/${result.video_id}`;
        video.addEventListener(
            'loadedmetadata',
            () => {
                video.currentTime = result.start_time ?? 0;
                video.pause();
            },
            { once: true }
        );
    }, [result]);

    const seekToSegment = useCallback(
        (index: number) => {
            const video = videoRef.current;
            if (index < 0 || index >= thumbnails.length) return;
            const seg = thumbnails[index];
            if (!video || !seg) return;
            video.currentTime = seg.start_time;
            video.play();
            setActiveIndex(index);
        },
        [thumbnails]
    );

    // Clamp activeIndex to valid range (guards against stale index after result change)
    const safeIndex = Math.min(activeIndex, thumbnails.length - 1);
    const activeSeg = thumbnails[safeIndex];

    if (!result) {
        return (
            <div className="flex items-center justify-center h-full bg-zinc-950 rounded-lg">
                <p className="text-zinc-500">点击结果卡片预览视频</p>
            </div>
        );
    }

    if (!activeSeg) {
        return null;
    }

    const showThumbnails = thumbnails.length > 1;

    return (
        <div className="flex gap-2">
            {/* Video element */}
            <div className="relative flex-1 min-w-0">
                <video ref={videoRef} controls className="w-full rounded-lg bg-black" />
                <div className="absolute top-2 left-2">
                    <Badge variant="secondary" className="font-mono text-xs bg-black/70 text-white border-0">
                        {formatTime(activeSeg.start_time)} -{' '}
                        {formatTime(activeSeg.end_time)}
                    </Badge>
                </div>
            </div>

            {/* Thumbnail column */}
            {showThumbnails && (
                <div className="flex flex-col gap-1.5 overflow-y-auto w-[48px] shrink-0">
                    {thumbnails.map((seg, i) => (
                        <button
                            key={i}
                            onClick={() => seekToSegment(i)}
                            className={`relative rounded overflow-hidden shrink-0 ${
                                i === safeIndex ? 'ring-2 ring-blue-500' : 'ring-1 ring-zinc-700'
                            }`}
                        >
                            <img
                                src={seg.keyframe_url}
                                alt={`Segment ${i + 1}`}
                                className="w-[48px] h-[36px] object-cover"
                            />
                            <span className="absolute bottom-0 left-0 right-0 text-[7px] leading-tight text-white text-center bg-black/70 py-px">
                                {formatTime(seg.start_time)}
                            </span>
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}
