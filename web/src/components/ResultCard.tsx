import type { SearchResult } from '../types';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { ScanSearch } from 'lucide-react';

interface ResultCardProps {
    result: SearchResult;
    isActive: boolean;
    onClick: () => void;
    onSimilar?: () => void;
    hideSimilar?: boolean;
}

function formatTime(seconds: number): string {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

export function ResultCard({ result, isActive, onClick, onSimilar, hideSimilar }: ResultCardProps) {
    const confidencePct = Math.round(result.confidence * 100);
    const matchTypes = result.match_type.split('+');

    return (
        <Card
            className={`cursor-pointer transition-all hover:bg-zinc-800/50 ${
                isActive ? 'ring-2 ring-blue-500 bg-zinc-800/50' : 'bg-zinc-900/50 border-zinc-800'
            }`}
            onClick={onClick}
        >
            <CardContent className="p-3 flex gap-3">
                {/* Keyframe thumbnail */}
                <div className="w-28 h-20 flex-shrink-0 rounded overflow-hidden bg-zinc-800">
                    {result.keyframe_url && (
                        <img src={result.keyframe_url} alt="" className="w-full h-full object-cover" loading="lazy" />
                    )}
                </div>

                {/* Info */}
                <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                        <span className="text-sm font-medium text-zinc-200 truncate">{result.filename}</span>
                        <span className="text-xs font-mono text-zinc-500 flex-shrink-0">
                            {formatTime(result.start_time ?? 0)} - {formatTime(result.end_time ?? 0)}
                        </span>
                    </div>

                    {/* Confidence bar */}
                    <div className="mt-1.5 flex items-center gap-2">
                        <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                            <div className="h-full bg-blue-500 rounded-full" style={{ width: `${confidencePct}%` }} />
                        </div>
                        <span className="text-xs font-mono text-zinc-400">{confidencePct}%</span>
                    </div>

                    {/* Match type badges */}
                    <div className="mt-1.5 flex gap-1 flex-wrap items-center">
                        {matchTypes.map((t) => (
                            <Badge key={t} variant="secondary" className="text-[10px] px-1.5 py-0 h-4">
                                {t}
                            </Badge>
                        ))}
                        {!hideSimilar && onSimilar && (
                            <button
                                className="ml-auto p-1 rounded hover:bg-zinc-700 text-zinc-500 hover:text-zinc-300 transition-colors"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onSimilar();
                                }}
                                title="相似片段"
                            >
                                <ScanSearch size={14} />
                            </button>
                        )}
                    </div>

                    {/* Description excerpt */}
                    {result.description && (
                        <p className="mt-1.5 text-xs text-zinc-500 line-clamp-2">{result.description}</p>
                    )}
                </div>
            </CardContent>
        </Card>
    );
}
