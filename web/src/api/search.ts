import type { SearchResponse } from '../types';

export async function search(
    query: string,
    page: number = 1,
    pageSize: number = 20,
    threshold?: number,
    dedup: boolean = true
): Promise<SearchResponse> {
    const params = new URLSearchParams({ q: query, page: String(page), page_size: String(pageSize) });
    if (threshold !== undefined) params.set('threshold', String(threshold));
    if (dedup) params.set('dedup', 'true');
    const res = await fetch(`/api/search?${params}`);
    if (!res.ok) throw new Error(`Search failed: ${res.statusText}`);
    return res.json();
}

export async function searchByImage(
    keyframeUrl: string,
    page: number = 1,
    pageSize: number = 20
): Promise<SearchResponse> {
    const params = new URLSearchParams({
        search_type: 'image',
        keyframe_search: keyframeUrl,
        page: String(page),
        page_size: String(pageSize),
        dedup: 'true',
    });
    const res = await fetch(`/api/search?${params}`);
    if (!res.ok) throw new Error(`Image search failed: ${res.statusText}`);
    return res.json();
}
