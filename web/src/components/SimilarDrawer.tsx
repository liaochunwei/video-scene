import { useState, useEffect } from 'react';
import type { SearchResult } from '../types';
import { ResultCard } from './ResultCard';
import { VideoPlayer } from './VideoPlayer';
import { X } from 'lucide-react';

interface SimilarDrawerProps {
    open: boolean;
    results: SearchResult[];
    loading: boolean;
    onClose: () => void;
}

export function SimilarDrawer({ open, results, loading, onClose }: SimilarDrawerProps) {
    const [activeResult, setActiveResult] = useState<SearchResult | null>(null);

    // Reset active result when results change (new search completed)
    useEffect(() => {
        setActiveResult(null);
    }, [results]);

    if (!open) {
        return null;
    }

    return (
        <>
            {/* Overlay */}
            <div className="fixed inset-0 bg-black/50 z-40" onClick={onClose} />
            {/* Drawer panel */}
            <div className="fixed right-0 top-0 bottom-0 w-[80%] bg-zinc-950 border-l border-zinc-800 z-50 flex flex-col animate-slide-in-right">
                {/* Header */}
                <div className="flex items-center justify-between p-3 border-b border-zinc-800">
                    <span className="text-sm font-medium text-zinc-200">相似片段</span>
                    <button
                        onClick={onClose}
                        className="p-1 rounded hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200"
                    >
                        <X size={18} />
                    </button>
                </div>
                {/* Body */}
                <div className="flex-1 flex overflow-hidden">
                    {/* Left: results list */}
                    <div className="w-[320px] shrink-0 overflow-y-auto p-3 space-y-2 border-r border-zinc-800">
                        {loading && <div className="text-center text-zinc-500 py-8">搜索中...</div>}
                        {!loading && results.length === 0 && (
                            <div className="text-center text-zinc-500 py-8">无相似片段</div>
                        )}
                        {results.map((r, i) => (
                            <ResultCard
                                key={`${r.video_id}-${r.start_time}-${i}`}
                                result={r}
                                isActive={
                                    activeResult?.video_id === r.video_id && activeResult?.start_time === r.start_time
                                }
                                onClick={() => setActiveResult(r)}
                                hideSimilar
                            />
                        ))}
                    </div>
                    {/* Right: video player */}
                    <div className="flex-1 p-4 flex items-start justify-center overflow-y-auto">
                        <div className="w-full max-w-[375px]">
                            <VideoPlayer result={activeResult} />
                            {activeResult && (
                                <div className="mt-3 space-y-1">
                                    <div className="text-sm font-medium text-zinc-200">{activeResult.filename}</div>
                                    <div className="text-xs text-zinc-500">{activeResult.description}</div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}
