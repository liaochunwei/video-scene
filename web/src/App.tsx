import { useState, useCallback } from 'react';
import type { SearchResult } from './types';
import { search, searchByImage } from './api/search';
import { SearchInput } from './components/SearchInput';
import { ResultCard } from './components/ResultCard';
import { VideoPlayer } from './components/VideoPlayer';
import { SimilarDrawer } from './components/SimilarDrawer';

const PAGE_SIZE = 20;

export default function App() {
    const [results, setResults] = useState<SearchResult[]>([]);
    const [total, setTotal] = useState(0);
    const [page, setPage] = useState(1);
    const [activeResult, setActiveResult] = useState<SearchResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [currentQuery, setCurrentQuery] = useState('');
    const [drawerOpen, setDrawerOpen] = useState(false);
    const [drawerResults, setDrawerResults] = useState<SearchResult[]>([]);
    const [drawerLoading, setDrawerLoading] = useState(false);
    const [dedup, setDedup] = useState(true);

    const handleSearch = useCallback(
        async (query: string) => {
            setLoading(true);
            setError(null);
            setCurrentQuery(query);
            try {
                const res = await search(query, 1, PAGE_SIZE, undefined, dedup);
                setResults(res.results);
                setTotal(res.total);
                setPage(1);
                setActiveResult(null);
            } catch (e) {
                setError(e instanceof Error ? e.message : 'Search failed');
            } finally {
                setLoading(false);
            }
        },
        [dedup]
    );

    const handlePageChange = useCallback(
        async (newPage: number) => {
            if (!currentQuery) return;
            setLoading(true);
            setError(null);
            try {
                const res = await search(currentQuery, newPage, PAGE_SIZE, undefined, dedup);
                setResults(res.results);
                setTotal(res.total);
                setPage(newPage);
            } catch (e) {
                setError(e instanceof Error ? e.message : 'Search failed');
            } finally {
                setLoading(false);
            }
        },
        [currentQuery, dedup]
    );

    const handleSimilarSearch = useCallback(async (result: SearchResult) => {
        setDrawerOpen(true);
        setDrawerLoading(true);
        setDrawerResults([]);
        try {
            const res = await searchByImage(result.keyframe_url);
            setDrawerResults(res.results);
        } catch (e) {
            console.error('Similar search failed:', e);
        } finally {
            setDrawerLoading(false);
        }
    }, []);

    const handleCloseDrawer = useCallback(() => {
        setDrawerOpen(false);
        setDrawerResults([]);
    }, []);

    const totalPages = Math.ceil(total / PAGE_SIZE);

    return (
        <div className="flex h-screen bg-zinc-950 text-zinc-100">
            {/* Left panel: search + results */}
            <div className="w-[420px] shrink-0 flex flex-col border-r border-zinc-800 overflow-hidden">
                <div className="p-3 border-b border-zinc-800">
                    <SearchInput onSearch={handleSearch} isLoading={loading} />
                    <label className="mt-3 p-3 flex items-center gap-2 text-xs text-zinc-400 cursor-pointer select-none">
                        <input
                            type="checkbox"
                            checked={dedup}
                            onChange={(e) => setDedup(e.target.checked)}
                            className="rounded border-zinc-600 bg-zinc-800"
                        />
                        合并同视频片段
                    </label>
                </div>
                <div className="flex-1 overflow-y-auto p-3 space-y-2">
                    {loading && results.length === 0 && <div className="text-center text-zinc-500 py-8">搜索中...</div>}
                    {error && <div className="text-center text-red-400 py-4 text-sm">{error}</div>}
                    {!loading && !error && results.length === 0 && currentQuery && (
                        <div className="text-center text-zinc-500 py-8">无搜索结果</div>
                    )}
                    {!currentQuery && <div className="text-center text-zinc-600 py-8 text-sm">输入搜索词开始搜索</div>}
                    {results.map((r, i) => (
                        <ResultCard
                            key={`${r.video_id}-${r.start_time}-${i}`}
                            result={r}
                            isActive={
                                activeResult?.video_id === r.video_id && activeResult?.start_time === r.start_time
                            }
                            onClick={() => setActiveResult(r)}
                            onSimilar={() => handleSimilarSearch(r)}
                        />
                    ))}
                </div>
                {/* Pagination */}
                {totalPages > 1 && (
                    <div className="p-3 border-t border-zinc-800 flex items-center justify-center gap-2">
                        <button
                            className="px-3 py-1 text-sm bg-zinc-800 rounded disabled:opacity-50"
                            disabled={page <= 1 || loading}
                            onClick={() => handlePageChange(page - 1)}
                        >
                            上一页
                        </button>
                        <span className="text-sm text-zinc-400">
                            {page} / {totalPages}
                        </span>
                        <button
                            className="px-3 py-1 text-sm bg-zinc-800 rounded disabled:opacity-50"
                            disabled={page >= totalPages || loading}
                            onClick={() => handlePageChange(page + 1)}
                        >
                            下一页
                        </button>
                    </div>
                )}
            </div>
            {/* Right panel: video player */}
            <div className="flex-1 p-4 flex items-start justify-center overflow-y-auto max-h-screen">
                <div className="w-full max-w-[480px]">
                    <VideoPlayer result={activeResult} paused={drawerOpen} />
                    {activeResult && (
                        <div className="mt-3 space-y-1">
                            <div className="text-sm font-medium text-zinc-200">{activeResult.filename}</div>
                            <div className="text-xs text-zinc-500">{activeResult.description}</div>
                        </div>
                    )}
                </div>
            </div>
            <SimilarDrawer
                open={drawerOpen}
                results={drawerResults}
                loading={drawerLoading}
                onClose={handleCloseDrawer}
            />
        </div>
    );
}
