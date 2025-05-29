import os
import time
import torch
import numa_bind
from collections import OrderedDict
from kvDiskSim import save_kvcache_memmap, load_kvcache_memmap


class LRUTieredKVCache:
    """
    LRU-based tiered KV cache system that manages KV cache across:
    - Local memory (fastest)
    - Remote memory (medium speed)
    - Disk storage (slowest)
    
    Uses LRU eviction policy to move items between tiers based on access patterns
    and memory thresholds.
    """
    
    def __init__(self, local_limit_mb=512, remote_limit_mb=1024, 
                 local_threshold=0.8, remote_threshold=0.8,
                 local_node=0, remote_node=1, kv_cache_dir='./kv_cache_disk/'):
        """
        Initialize the tiered cache system.
        
        Args:
            local_limit_mb: Maximum local memory in MB
            remote_limit_mb: Maximum remote memory in MB  
            local_threshold: Threshold (0-1) to trigger eviction from local to remote
            remote_threshold: Threshold (0-1) to trigger eviction from remote to disk
            local_node: NUMA node for local memory
            remote_node: NUMA node for remote memory
            kv_cache_dir: Directory for disk storage
        """
        self.local_limit_mb = local_limit_mb
        self.remote_limit_mb = remote_limit_mb
        self.local_threshold = local_threshold
        self.remote_threshold = remote_threshold
        self.local_node = local_node
        self.remote_node = remote_node
        self.kv_cache_dir = kv_cache_dir
        
        # Storage for each tier - using OrderedDict for LRU tracking
        # Key format: (request_id, layer_id)
        self.local_cache = OrderedDict()  # Most recently used at end
        self.remote_cache = OrderedDict()
        self.disk_cache = set()  # Just track which keys are on disk
        
        # Size tracking in MB
        self.local_size_mb = 0.0
        self.remote_size_mb = 0.0
        
        # Setup disk directory
        os.makedirs(self.kv_cache_dir, exist_ok=True)
        
        # Track current NUMA binding
        self.current_numa_node = local_node
        numa_bind.set_membind(local_node)
    
    def _get_tensor_size_mb(self, kv_tuple):
        """Calculate size of KV tuple in MB."""
        if kv_tuple is None:
            return 0.0
        keys, values = kv_tuple
        size_bytes = (keys.element_size() * keys.numel() + 
                     values.element_size() * values.numel())
        return size_bytes / (1024 ** 2)
    
    def _move_to_numa_node(self, target_node):
        """Switch NUMA binding if needed."""
        if self.current_numa_node != target_node:
            numa_bind.set_membind(target_node)
            self.current_numa_node = target_node
    
    def _evict_from_local_to_remote(self, device):
        """Evict LRU items from local to remote memory."""
        print(f"Evicting from local to remote memory...")
        self._move_to_numa_node(self.remote_node)
        
        # Evict until we're below threshold
        target_size = self.local_limit_mb * self.local_threshold * 0.9  # Leave some buffer
        
        while self.local_size_mb > target_size and self.local_cache:
            # Get least recently used item (first in OrderedDict)
            key, kv_tuple = self.local_cache.popitem(last=False)
            size_mb = self._get_tensor_size_mb(kv_tuple)
            
            # Move to remote memory
            if kv_tuple is not None:
                # Ensure tensors are on correct device and NUMA node
                keys, values = kv_tuple
                keys = keys.to(device)
                values = values.to(device)
                self.remote_cache[key] = (keys, values)
                self.remote_size_mb += size_mb
            
            self.local_size_mb -= size_mb
            print(f"Moved {key} from local to remote. Local: {self.local_size_mb:.2f}MB")
        
        self._move_to_numa_node(self.local_node)
    
    def _evict_from_remote_to_disk(self):
        """Evict LRU items from remote to disk."""
        print(f"Evicting from remote to disk...")
        
        # Evict until we're below threshold
        target_size = self.remote_limit_mb * self.remote_threshold * 0.9
        
        while self.remote_size_mb > target_size and self.remote_cache:
            # Get least recently used item
            key, kv_tuple = self.remote_cache.popitem(last=False)
            size_mb = self._get_tensor_size_mb(kv_tuple)
            
            # Save to disk
            request_id, layer_id = key
            if kv_tuple is not None:
                save_kvcache_memmap(request_id, layer_id, kv_tuple, self.kv_cache_dir)
                self.disk_cache.add(key)
            
            self.remote_size_mb -= size_mb
            print(f"Moved {key} from remote to disk. Remote: {self.remote_size_mb:.2f}MB")
    
    def get(self, request_id, layer_id, device):
        """
        Get KV cache for a specific request and layer.
        Implements LRU promotion - accessed items move to end of OrderedDict.
        """
        key = (request_id, layer_id)
        
        # Check local cache first (fastest)
        if key in self.local_cache:
            # Move to end (mark as recently used)
            kv_tuple = self.local_cache.pop(key)
            self.local_cache[key] = kv_tuple
            print(f"Cache hit: local memory for {key}")
            return kv_tuple
        
        # Check remote cache
        if key in self.remote_cache:
            # Move to end and promote to local if there's space
            self._move_to_numa_node(self.remote_node)
            kv_tuple = self.remote_cache.pop(key)
            size_mb = self._get_tensor_size_mb(kv_tuple)
            self.remote_size_mb -= size_mb
            
            # Try to promote to local memory
            if self.local_size_mb + size_mb <= self.local_limit_mb * self.local_threshold:
                self._move_to_numa_node(self.local_node)
                if kv_tuple is not None:
                    keys, values = kv_tuple
                    keys = keys.to(device)
                    values = values.to(device)
                    self.local_cache[key] = (keys, values)
                    self.local_size_mb += size_mb
                    print(f"Cache hit: promoted {key} from remote to local")
                    return (keys, values)
            else:
                # Keep in remote but mark as recently used
                self.remote_cache[key] = kv_tuple
                self.remote_size_mb += size_mb
                print(f"Cache hit: remote memory for {key}")
                self._move_to_numa_node(self.local_node)
                return kv_tuple
        
        # Check disk cache
        if key in self.disk_cache:
            # Load from disk
            request_id, layer_id = key
            kv_tuple = load_kvcache_memmap(request_id, layer_id, self.kv_cache_dir, device)
            if kv_tuple is not None:
                size_mb = self._get_tensor_size_mb(kv_tuple)
                self.disk_cache.remove(key)
                
                # Try to promote to remote memory first
                if self.remote_size_mb + size_mb <= self.remote_limit_mb * self.remote_threshold:
                    self._move_to_numa_node(self.remote_node)
                    keys, values = kv_tuple
                    keys = keys.to(device)
                    values = values.to(device)
                    self.remote_cache[key] = (keys, values)
                    self.remote_size_mb += size_mb
                    print(f"Cache hit: promoted {key} from disk to remote")
                    self._move_to_numa_node(self.local_node)
                    return (keys, values)
                else:
                    # If remote is full, keep on disk but still return the data
                    self.disk_cache.add(key)  # Put back in disk cache
                    print(f"Cache hit: disk storage for {key} (remote full)")
                    return kv_tuple
            
        # Cache miss - return None
        print(f"Cache miss for {key}")
        return None
    
    def put(self, request_id, layer_id, kv_tuple, device):
        """
        Store KV cache, managing evictions as needed.
        """
        key = (request_id, layer_id)
        size_mb = self._get_tensor_size_mb(kv_tuple)
        
        # Remove from any existing location first
        self._remove_key(key)
        
        # Try to store in local memory first
        if self.local_size_mb + size_mb <= self.local_limit_mb * self.local_threshold:
            self._move_to_numa_node(self.local_node)
            self.local_cache[key] = kv_tuple
            self.local_size_mb += size_mb
            print(f"Stored {key} in local memory. Size: {self.local_size_mb:.2f}MB")
            return
        
        # Local is full, try eviction
        if self.local_cache:
            self._evict_from_local_to_remote(device)
            
            # Try again after eviction
            if self.local_size_mb + size_mb <= self.local_limit_mb * self.local_threshold:
                self._move_to_numa_node(self.local_node)
                self.local_cache[key] = kv_tuple
                self.local_size_mb += size_mb
                print(f"Stored {key} in local memory after eviction. Size: {self.local_size_mb:.2f}MB")
                return
        
        # Store in remote memory
        if self.remote_size_mb + size_mb <= self.remote_limit_mb * self.remote_threshold:
            self._move_to_numa_node(self.remote_node)
            if kv_tuple is not None:
                keys, values = kv_tuple
                keys = keys.to(device)
                values = values.to(device)
                self.remote_cache[key] = (keys, values)
                self.remote_size_mb += size_mb
                print(f"Stored {key} in remote memory. Size: {self.remote_size_mb:.2f}MB")
                self._move_to_numa_node(self.local_node)
                return
        
        # Remote is full, try eviction
        if self.remote_cache:
            self._evict_from_remote_to_disk()
            
            # Try again after eviction
            if self.remote_size_mb + size_mb <= self.remote_limit_mb * self.remote_threshold:
                self._move_to_numa_node(self.remote_node)
                if kv_tuple is not None:
                    keys, values = kv_tuple
                    keys = keys.to(device)
                    values = values.to(device)
                    self.remote_cache[key] = (keys, values)
                    self.remote_size_mb += size_mb
                    print(f"Stored {key} in remote memory after eviction. Size: {self.remote_size_mb:.2f}MB")
                    self._move_to_numa_node(self.local_node)
                    return
        
        # Store on disk as last resort
        if kv_tuple is not None:
            save_kvcache_memmap(request_id, layer_id, kv_tuple, self.kv_cache_dir)
            self.disk_cache.add(key)
            print(f"Stored {key} on disk")
    
    def _remove_key(self, key):
        """Remove key from all tiers if it exists."""
        if key in self.local_cache:
            kv_tuple = self.local_cache.pop(key)
            self.local_size_mb -= self._get_tensor_size_mb(kv_tuple)
        
        if key in self.remote_cache:
            kv_tuple = self.remote_cache.pop(key)
            self.remote_size_mb -= self._get_tensor_size_mb(kv_tuple)
        
        if key in self.disk_cache:
            self.disk_cache.remove(key)
            # Note: We don't delete the actual disk files for simplicity
            # They can be cleaned up later
    
    def get_stats(self):
        """Get current cache statistics."""
        return {
            "local_size_mb": self.local_size_mb,
            "remote_size_mb": self.remote_size_mb,
            "local_count": len(self.local_cache),
            "remote_count": len(self.remote_cache),
            "disk_count": len(self.disk_cache),
            "local_utilization": self.local_size_mb / self.local_limit_mb,
            "remote_utilization": self.remote_size_mb / self.remote_limit_mb,
        }
    
    def cleanup(self):
        """Clean up disk cache files."""
        if os.path.isdir(self.kv_cache_dir) and "kv_cache_disk" in self.kv_cache_dir:
            for root, _, files in os.walk(self.kv_cache_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                if root != self.kv_cache_dir:  # Don't remove the base directory
                    os.rmdir(root)
        
        # Reset NUMA binding to local
        self._move_to_numa_node(self.local_node) 