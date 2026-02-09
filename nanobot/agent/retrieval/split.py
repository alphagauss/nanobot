import re


def split_markdown_into_chunks(content: str, max_chunk_size: int = 1000) -> list[str]:
    """
    Split markdown content into semantic chunks.
    Prioritizes splitting by headers, then by double newlines.
    """
    if not content:
        return []

    # Split by headers (h1, h2, h3) but keep the headers with the content
    # This regex looks for a newline followed by a header
    raw_chunks = re.split(r'\n(?=#+ )', content)
    
    final_chunks = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
            
        # If a chunk is still too large, split it by double newlines
        if len(chunk) > max_chunk_size:
            sub_chunks = chunk.split('\n\n')
            current_sub_chunk = ""
            
            for sub in sub_chunks:
                if len(current_sub_chunk) + len(sub) < max_chunk_size:
                    current_sub_chunk += ("\n\n" if current_sub_chunk else "") + sub
                else:
                    if current_sub_chunk:
                        final_chunks.append(current_sub_chunk.strip())
                    current_sub_chunk = sub
            
            if current_sub_chunk:
                final_chunks.append(current_sub_chunk.strip())
        else:
            final_chunks.append(chunk)
            
    return final_chunks