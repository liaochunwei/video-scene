import { useState } from 'react';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { Search } from 'lucide-react';

interface SearchInputProps {
    onSearch: (query: string) => void;
    isLoading?: boolean;
}

export function SearchInput({ onSearch, isLoading }: SearchInputProps) {
    const [query, setQuery] = useState('');

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (query.trim()) onSearch(query.trim());
    };

    return (
        <form onSubmit={handleSubmit} className="flex gap-2">
            <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="输入搜索词..."
                className="flex-1"
            />
            <Button type="submit" disabled={isLoading || !query.trim()}>
                <Search className="h-4 w-4" />
            </Button>
        </form>
    );
}
