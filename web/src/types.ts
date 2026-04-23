export interface MoreSegment {
    segment_id: string;
    start_time?: number;
    end_time?: number;
    confidence: number;
    keyframe_url: string;
}

export interface SearchResult {
    video_id: string;
    filename: string;
    start_time?: number;
    end_time?: number;
    keyframe_url: string;
    confidence: number;
    confidence_low: number;
    confidence_high: number;
    match_type: string;
    match_label: string;
    description: string;
    more: MoreSegment[];
}

export interface SearchResponse {
    total: number;
    page: number;
    page_size: number;
    results: SearchResult[];
}
