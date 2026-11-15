#pragma once

enum class MemMode { Pageable, Pinned, Managed, ManagedPrefetch, PageableDirect };

inline const char* to_string(MemMode m) {
    switch (m) {
        case MemMode::Pageable:        return "pageable";
        case MemMode::Pinned:          return "pinned";
        case MemMode::Managed:         return "managed";
        case MemMode::ManagedPrefetch: return "managed_prefetch";
        case MemMode::PageableDirect:  return "pageable_direct";
    }
    return "unknown";
}
